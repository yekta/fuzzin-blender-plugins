bl_info = {
    "name": "Fuzzin Pipeline",
    "author": "Yekta",
    "version": (5, 3, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Fuzzin Pipeline",
    "description": "Scale to Height, Flatten Bottom, Bottom Female Connector, Feature Connectors, and Left/Right Marking in one unified workflow",
    "category": "Mesh",
}

import bpy
import bmesh
import math
import os
from mathutils import Matrix, Vector
from collections import deque
from bpy.props import (
    FloatProperty,
    EnumProperty,
    FloatVectorProperty,
    BoolProperty,
    IntProperty,
)


# ===========================================================================
# Shared BFS Flood Fill
# ===========================================================================


def bfs_feature_fill(bm, seed_indices, grad_limit_rad):
    """Flood-fill from seed vertices, stopping when the normal angle between
    neighbours exceeds *grad_limit_rad*.

    Returns (selected: set[int], boundary: set[int]).
    """
    visited = set()
    selected = set()
    boundary = set()
    queue = deque()

    for idx in seed_indices:
        if idx < len(bm.verts):
            v = bm.verts[idx]
            visited.add(v.index)
            selected.add(v.index)
            queue.append(v)

    while queue:
        v = queue.popleft()
        for e in v.link_edges:
            other = e.other_vert(v)
            if other.index in visited:
                continue
            visited.add(other.index)

            if other.is_boundary or other.is_wire:
                boundary.add(other.index)
                continue

            if v.normal.angle(other.normal, 0.0) > grad_limit_rad:
                boundary.add(other.index)
            else:
                selected.add(other.index)
                queue.append(other)

    return selected, boundary


def detect_optimal_angle(bm, seed_indices, angle_min, angle_max):
    """Scan integer angles in [angle_min, angle_max] and find the breakpoint
    where the vertex count jumps the most.  Returns the angle just *before*
    the biggest jump (i.e. the last angle that still captures only the feature).

    Returns (optimal_angle_deg: float, scan_data: list[tuple[int, int]])
    where scan_data is [(angle, vert_count), ...] for debug / reporting.
    """
    scan = []
    for deg in range(int(math.ceil(angle_min)), int(math.floor(angle_max)) + 1):
        rad = math.radians(deg)
        sel, bnd = bfs_feature_fill(bm, seed_indices, rad)
        total = len(sel | bnd)
        scan.append((deg, total))

    if len(scan) < 2:
        mid = (angle_min + angle_max) / 2.0
        return mid, scan

    max_jump = 0
    break_idx = 1
    for i in range(1, len(scan)):
        jump = scan[i][1] - scan[i - 1][1]
        if jump > max_jump:
            max_jump = jump
            break_idx = i

    optimal_deg = float(scan[break_idx - 1][0])
    return optimal_deg, scan


# ===========================================================================
# Bottom Flattening — Boolean Cut
# ===========================================================================


def detect_bottom_cut_level(obj, zone_height):
    """Analyse the mesh to find the ideal Z level for a flat bottom cut.

    Strategy
    --------
    1. Find the global minimum Z in world space.
    2. Collect every vertex whose Z is within *zone_height* of that minimum —
       these define the "bottom zone".
    3. The cut level is the *maximum* Z among those vertices.  Cutting here
       guarantees that every surface in the bottom zone becomes co-planar.

    Returns (cut_z, min_z, bottom_vert_count).
    """
    me = obj.data
    world = obj.matrix_world

    zs = [(world @ v.co).z for v in me.vertices]
    if not zs:
        return 0.0, 0.0, 0

    min_z = min(zs)
    bottom_zs = [z for z in zs if z - min_z <= zone_height]
    cut_z = max(bottom_zs) if bottom_zs else min_z

    return cut_z, min_z, len(bottom_zs)


def create_flatten_cutter_obj(context, obj, cut_z):
    """Build a large box whose top face sits exactly at *cut_z*.

    The box extends far beyond the model in XY and well below it in Z so that
    a Boolean DIFFERENCE removes everything below the cut plane, leaving a
    perfectly flat bottom surface.
    """
    me = obj.data
    world = obj.matrix_world

    verts_world = [world @ v.co for v in me.vertices]
    xs = [v.x for v in verts_world]
    ys = [v.y for v in verts_world]
    zs = [v.z for v in verts_world]

    # Generous margin so the cutter fully envelops the model in XY
    bbox_diag = math.sqrt(
        (max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2 + (max(zs) - min(zs)) ** 2
    )
    margin = bbox_diag * 1.5

    x_min = min(xs) - margin
    x_max = max(xs) + margin
    y_min = min(ys) - margin
    y_max = max(ys) + margin
    z_min = min(zs) - margin  # well below the model
    z_max = cut_z  # top face = the cut plane

    # Build box from 8 corners
    bm = bmesh.new()
    v0 = bm.verts.new(Vector((x_min, y_min, z_min)))
    v1 = bm.verts.new(Vector((x_max, y_min, z_min)))
    v2 = bm.verts.new(Vector((x_max, y_max, z_min)))
    v3 = bm.verts.new(Vector((x_min, y_max, z_min)))
    v4 = bm.verts.new(Vector((x_min, y_min, z_max)))
    v5 = bm.verts.new(Vector((x_max, y_min, z_max)))
    v6 = bm.verts.new(Vector((x_max, y_max, z_max)))
    v7 = bm.verts.new(Vector((x_min, y_max, z_max)))

    bm.faces.new([v3, v2, v1, v0])  # bottom
    bm.faces.new([v4, v5, v6, v7])  # top (cut plane)
    bm.faces.new([v0, v1, v5, v4])  # front
    bm.faces.new([v2, v3, v7, v6])  # back
    bm.faces.new([v3, v0, v4, v7])  # left
    bm.faces.new([v1, v2, v6, v5])  # right

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    mesh = bpy.data.meshes.new("_FlattenCutter")
    bm.to_mesh(mesh)
    bm.free()

    cutter = bpy.data.objects.new("_FlattenCutter", mesh)
    context.collection.objects.link(cutter)
    return cutter


# ===========================================================================
# Octagon Cutter for Female Connector
# ===========================================================================

# Fixed bottom-connector dimensions (mm)
_BOTTOM_CONN_WIDTH = 6.3  # flat-to-flat across the octagon
_BOTTOM_CONN_DEPTH = 6.4  # pocket depth (Z)
_BOTTOM_CONN_FILLET = 0.4  # fillet radius on vertical (side) edges
_BOTTOM_CONN_FILLET_SEGS = 64  # segments per fillet arc
_BOTTOM_CONN_CHAMFER = 0.4  # 45-degree entry chamfer


def create_octagon_cutter_bm(
    width=_BOTTOM_CONN_WIDTH,
    depth=_BOTTOM_CONN_DEPTH,
    fillet_radius=_BOTTOM_CONN_FILLET,
    segments=_BOTTOM_CONN_FILLET_SEGS,
    entry_chamfer=_BOTTOM_CONN_CHAMFER,
):
    """Build a bmesh octagonal prism with filleted vertical edges and an
    optional entry chamfer.

    The cross-section is a regular octagon whose flat-to-flat distance equals
    *width*.  It is extruded along Z from 0 to *depth*.

    *fillet_radius* rounds the eight vertical (side) edges.  Each fillet arc
    is sampled with *segments* points.

    If *entry_chamfer* > 0 the bottom of the cutter (the pocket opening) is
    flared outward by that amount over a 45-degree slope — exactly like
    applying a chamfer to the entry edges in a CAD tool after filleting.

    Normals point outward so this can be used as a boolean cutter.
    """
    bm = bmesh.new()

    n_sides = 8
    apothem = width / 2.0  # centre → flat
    R = apothem / math.cos(math.pi / n_sides)  # circumradius

    # Clamp fillet so arcs don't overlap on adjacent corners
    side_len = 2.0 * R * math.sin(math.pi / n_sides)
    max_fillet = side_len / 2.0 - 0.001
    r = max(min(fillet_radius, max_fillet), 0.0)

    # Each corner of a regular octagon has an interior angle of 135°.
    # The fillet arc sweeps 180° − 135° = 45° (= 2π / n_sides).
    arc_sweep = 2.0 * math.pi / n_sides  # radians per corner

    def make_profile(z, offset=0.0):
        """Create a ring of vertices for one Z-level.

        *offset* uniformly expands the profile outward (used for the
        chamfer flare at the entry).
        """
        verts = []
        apothem_eff = apothem + offset
        r_eff = r + offset
        # Arc-centre polygon circumradius
        R_c = (
            (apothem_eff - r_eff) / math.cos(math.pi / n_sides)
            if (apothem_eff - r_eff) > 0
            else 0.0
        )

        for corner in range(n_sides):
            # Vertex angle of the un-filleted corner
            theta = math.pi / n_sides + corner * (2.0 * math.pi / n_sides)
            # Arc centre
            cx = R_c * math.cos(theta)
            cy = R_c * math.sin(theta)
            # Arc runs from θ − sweep/2 to θ + sweep/2
            arc_start = theta - arc_sweep / 2.0
            for i in range(segments):
                a = arc_start + i * arc_sweep / segments
                x = cx + r_eff * math.cos(a)
                y = cy + r_eff * math.sin(a)
                verts.append(bm.verts.new(Vector((x, y, z))))
        return verts

    # Clamp chamfer so it doesn't exceed the pocket depth
    cham = max(min(entry_chamfer, depth * 0.5), 0.0)

    # Build rings -------------------------------------------------------
    # When chamfer is active the cutter has three levels:
    #   bottom (z=0)   : flared profile  – the widened opening
    #   chamfer (z=cham): nominal profile – where the 45° slope ends
    #   top (z=depth)  : nominal profile – ceiling of the pocket
    # Without chamfer it's just the usual two-ring extrusion.

    if cham > 1e-6:
        bottom_ring = make_profile(0.0, offset=cham)  # flared
        chamfer_ring = make_profile(cham, offset=0.0)  # nominal
        top_ring = make_profile(depth, offset=0.0)  # nominal
    else:
        bottom_ring = make_profile(0.0)
        chamfer_ring = None
        top_ring = make_profile(depth)

    bm.verts.ensure_lookup_table()

    n = len(bottom_ring)

    # --- cap faces ---
    try:
        bm.faces.new(list(reversed(bottom_ring)))
    except ValueError:
        pass
    try:
        bm.faces.new(top_ring)
    except ValueError:
        pass

    if chamfer_ring is not None:
        # Side quads: bottom_ring -> chamfer_ring (the 45° chamfer band)
        for i in range(n):
            j = (i + 1) % n
            try:
                bm.faces.new(
                    [bottom_ring[i], bottom_ring[j], chamfer_ring[j], chamfer_ring[i]]
                )
            except ValueError:
                pass
        # Side quads: chamfer_ring -> top_ring (the straight walls)
        for i in range(n):
            j = (i + 1) % n
            try:
                bm.faces.new(
                    [chamfer_ring[i], chamfer_ring[j], top_ring[j], top_ring[i]]
                )
            except ValueError:
                pass
    else:
        # Side quads: bottom_ring -> top_ring (no chamfer)
        for i in range(n):
            j = (i + 1) % n
            try:
                bm.faces.new([bottom_ring[i], bottom_ring[j], top_ring[j], top_ring[i]])
            except ValueError:
                pass

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


# ===========================================================================
# Left / Right T-Mark Cutter
# ===========================================================================

# Fixed mark dimensions (mm)
_TMARK_HEIGHT = 3.0  # total height of the mark
_TMARK_WIDTH = 2.0  # width of the horizontal bar
_TMARK_LINE = 0.4  # stroke thickness of the lines
_TMARK_DEPTH = 0.8  # cut depth into the part


def _tmark_profile(side="LEFT"):
    """Return a list of (y, z) vertices tracing the outline of a side mark,
    centred at y=0, z=0 (midpoint of the total height).

    Both shapes are horizontally mirrored so they read correctly when
    viewed from the back of the part (looking in the +X direction).
    The full bounding box of each shape is centred at y=0 so the mark
    appears visually centred on the back face.

    LEFT  – capital "L": vertical stem + bar at bottom.
    RIGHT – capital "T": vertical stem + bar at top, centred.
    """
    hw = _TMARK_WIDTH / 2.0  # half-width extent
    hs = _TMARK_LINE / 2.0  # half stem thickness
    h = _TMARK_HEIGHT
    t = _TMARK_LINE  # stroke thickness

    bot = -h / 2.0
    top = h / 2.0

    if side == "LEFT":
        # L mirrored: stem + foot at bottom extending right (toward -Y)
        raw = [
            (hs, top),
            (-hs, top),
            (-hs, bot + t),
            (-hw, bot + t),
            (-hw, bot),
            (hs, bot),
        ]
    else:
        # T: horizontal bar at top spanning full width, stem drops down
        raw = [
            (hw, top),  # bar top-right
            (hw, top - t),  # bar bottom-right
            (hs, top - t),  # bar meets stem right
            (hs, bot),  # stem bottom-right
            (-hs, bot),  # stem bottom-left
            (-hs, top - t),  # bar meets stem left
            (-hw, top - t),  # bar bottom-left
            (-hw, top),  # bar top-left
        ]

    # Centre the full shape width at y=0
    y_min = min(y for y, z in raw)
    y_max = max(y for y, z in raw)
    y_shift = -(y_min + y_max) / 2.0
    return [(y + y_shift, z) for y, z in raw]


def create_tmark_cutter_bm(side="LEFT"):
    """Build a bmesh cutter for an L or r shaped engraving mark.

    The profile lies on the YZ plane.  The cutter extends in +X from 0
    to *_TMARK_DEPTH* (plus a small overlap for clean booleans).

    *side*: ``"LEFT"`` for capital L, ``"RIGHT"`` for lowercase r.
    """
    bm = bmesh.new()
    profile = _tmark_profile(side)
    n = len(profile)
    overlap = 0.05  # small extra to avoid co-planar boolean issues
    x_front = -overlap
    x_back = _TMARK_DEPTH + overlap

    front_verts = []
    back_verts = []
    for y, z in profile:
        front_verts.append(bm.verts.new(Vector((x_front, y, z))))
        back_verts.append(bm.verts.new(Vector((x_back, y, z))))

    bm.verts.ensure_lookup_table()

    # Front face (winding for outward -X normal)
    try:
        bm.faces.new(front_verts)
    except ValueError:
        pass

    # Back face (reversed winding for outward +X normal)
    try:
        bm.faces.new(list(reversed(back_verts)))
    except ValueError:
        pass

    # Side quads connecting front and back rings
    for i in range(n):
        j = (i + 1) % n
        try:
            bm.faces.new([front_verts[i], front_verts[j], back_verts[j], back_verts[i]])
        except ValueError:
            pass

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def _mark_rotation_matrix(direction):
    """Return a 4×4 rotation matrix that transforms the default cutter
    orientation (profile on YZ, extending +X) to cut inward from the
    face specified by *direction*.
    """
    pi = math.pi
    rotations = {
        # Axis-aligned: rotate +X to point inward (= -direction_vec)
        "NEG_X": Matrix.Identity(4),  # +X → +X (inward from -X face)
        "POS_X": Matrix.Rotation(pi, 4, "Z"),  # +X → -X
        "NEG_Y": Matrix.Rotation(pi / 2, 4, "Z"),  # +X → +Y
        "POS_Y": Matrix.Rotation(-pi / 2, 4, "Z"),  # +X → -Y
        "NEG_Z": Matrix.Rotation(-pi / 2, 4, "Y"),  # +X → +Z
        "POS_Z": Matrix.Rotation(pi / 2, 4, "Y"),  # +X → -Z
        # XY diagonals: rotate around Z
        "NEG_X_NEG_Y": Matrix.Rotation(pi / 4, 4, "Z"),  # +X → (+X+Y)/√2
        "NEG_X_POS_Y": Matrix.Rotation(-pi / 4, 4, "Z"),  # +X → (+X-Y)/√2
        "POS_X_NEG_Y": Matrix.Rotation(3 * pi / 4, 4, "Z"),  # +X → (-X+Y)/√2
        "POS_X_POS_Y": Matrix.Rotation(-3 * pi / 4, 4, "Z"),  # +X → (-X-Y)/√2
        # XZ diagonals: rotate around Y
        "NEG_X_NEG_Z": Matrix.Rotation(-pi / 4, 4, "Y"),  # +X → (+X+Z)/√2
        "NEG_X_POS_Z": Matrix.Rotation(pi / 4, 4, "Y"),  # +X → (+X-Z)/√2
        "POS_X_NEG_Z": Matrix.Rotation(-3 * pi / 4, 4, "Y"),  # +X → (-X+Z)/√2
        "POS_X_POS_Z": Matrix.Rotation(3 * pi / 4, 4, "Y"),  # +X → (-X-Z)/√2
        # YZ diagonals: first rotate to ±Y axis, then tilt ±45° around X
        "NEG_Y_NEG_Z": Matrix.Rotation(pi / 4, 4, "X")
        @ Matrix.Rotation(pi / 2, 4, "Z"),
        "NEG_Y_POS_Z": Matrix.Rotation(-pi / 4, 4, "X")
        @ Matrix.Rotation(pi / 2, 4, "Z"),
        "POS_Y_NEG_Z": Matrix.Rotation(-pi / 4, 4, "X")
        @ Matrix.Rotation(-pi / 2, 4, "Z"),
        "POS_Y_POS_Z": Matrix.Rotation(pi / 4, 4, "X")
        @ Matrix.Rotation(-pi / 2, 4, "Z"),
    }
    return rotations.get(direction, Matrix.Identity(4))


def _find_mesh_islands(obj):
    """Detect disconnected mesh islands inside *obj* using BFS on edges.

    Returns a list of islands, each island being a set of vertex indices
    (in local/mesh space).
    """
    import bmesh as _bm

    me = obj.data
    was_edit = obj.mode == "EDIT"

    bm = _bm.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()

    visited = set()
    islands = []

    for v in bm.verts:
        if v.index in visited:
            continue
        # BFS from this vertex
        island = set()
        queue = deque([v])
        while queue:
            cur = queue.popleft()
            if cur.index in visited:
                continue
            visited.add(cur.index)
            island.add(cur.index)
            for e in cur.link_edges:
                other = e.other_vert(cur)
                if other.index not in visited:
                    queue.append(other)
        if island:
            islands.append(island)

    bm.free()
    return islands


def _island_back_face_centre(obj, vert_indices):
    """For a subset of vertex indices (one island), find the placement point
    for the mark on the back (-X) face.

    Collects all vertices near the minimum X (within a tolerance) and
    averages their Y and Z — effectively the centre of gravity of the
    back plane.

    Returns (min_x, centre_y, centre_z).
    """
    me = obj.data
    world = obj.matrix_world

    coords = [(vi, world @ me.vertices[vi].co) for vi in vert_indices]
    if not coords:
        return 0.0, 0.0, 0.0

    min_x = min(c.x for _, c in coords)

    # Tolerance: gather all verts that form the back plane
    tol = 0.2  # mm
    back = [c for _, c in coords if c.x - min_x <= tol]

    centre_y = sum(v.y for v in back) / len(back)
    centre_z = sum(v.z for v in back) / len(back)

    return min_x, centre_y, centre_z


def _island_face_centre(obj, vert_indices, direction="NEG_X"):
    """Find the placement point for the mark on the face determined by *direction*.

    Looks for the flat plane perpendicular to *direction* that sits farthest
    along it: collect polygons whose outward normal is aligned with *dir_vec*
    (within ~15°), then pick the coplanar group whose centers share the
    maximum projection along *dir_vec*, and return their area-weighted
    centroid in world space.

    Falls back to the vertex-extremum method when no aligned faces exist
    (e.g. the island is fully curved or beveled).
    """
    dir_vec = _parse_direction(direction)
    me = obj.data
    world = obj.matrix_world
    # Inverse-transpose handles non-uniform scale when rotating normals.
    normal_mat = world.to_3x3().inverted().transposed()

    vert_set = set(vert_indices)
    align_threshold = math.cos(math.radians(15.0))

    aligned = []  # (world_center, proj_along_dir, area)
    for poly in me.polygons:
        if not all(vi in vert_set for vi in poly.vertices):
            continue
        nrm = (normal_mat @ poly.normal).normalized()
        if nrm.dot(dir_vec) < align_threshold:
            continue
        center = world @ poly.center
        aligned.append((center, dir_vec.dot(center), poly.area))

    if aligned:
        extremum = max(p for _, p, _ in aligned)
        plane_tol = 0.1  # mm — group of coplanar faces at the extremum
        group = [(c, a) for c, p, a in aligned if abs(p - extremum) <= plane_tol]
        total_area = sum(a for _, a in group)
        if total_area > 0.0:
            cx = sum(c.x * a for c, a in group) / total_area
            cy = sum(c.y * a for c, a in group) / total_area
            cz = sum(c.z * a for c, a in group) / total_area
            return Vector((cx, cy, cz))

    # Fallback: no face aligned with the requested direction.
    coords = [world @ me.vertices[vi].co for vi in vert_indices]
    if not coords:
        return Vector((0, 0, 0))
    projs = [dir_vec.dot(c) for c in coords]
    extremum_proj = max(projs)
    tol = 0.2  # mm
    face_verts = [c for c, p in zip(coords, projs) if abs(p - extremum_proj) <= tol]
    if not face_verts:
        face_verts = coords
    cx = sum(v.x for v in face_verts) / len(face_verts)
    cy = sum(v.y for v in face_verts) / len(face_verts)
    cz = sum(v.z for v in face_verts) / len(face_verts)
    return Vector((cx, cy, cz))


def _island_centroid(obj, vert_indices):
    """Return the world-space centroid (x, y, z) of a set of vertex indices."""
    me = obj.data
    world = obj.matrix_world
    coords = [world @ me.vertices[vi].co for vi in vert_indices]
    n = len(coords)
    if n == 0:
        return Vector((0, 0, 0))
    return Vector(
        (
            sum(c.x for c in coords) / n,
            sum(c.y for c in coords) / n,
            sum(c.z for c in coords) / n,
        )
    )


# ===========================================================================
# Property Groups
# ===========================================================================


class CPIPE_Props(bpy.types.PropertyGroup):
    # --- Scale to Height ---
    scale_enabled: BoolProperty(
        name="Scale to Height",
        description="Scale the model so the reference distance matches the target height",
        default=False,
    )
    target_height_mm: FloatProperty(
        name="Target Height (mm)",
        description="Desired distance between the two reference points in millimetres",
        default=60.2,
        min=0.001,
        soft_min=1.0,
        soft_max=1000.0,
        precision=2,
        unit="NONE",
    )
    point_a: FloatVectorProperty(subtype="XYZ")
    point_b: FloatVectorProperty(subtype="XYZ")
    ref_distance: FloatProperty(default=0.0)
    scale_points_set: BoolProperty(default=False)

    # --- Scale to Height: Previous (for restore) ---
    prev_point_a: FloatVectorProperty(subtype="XYZ")
    prev_point_b: FloatVectorProperty(subtype="XYZ")
    prev_ref_distance: FloatProperty(default=0.0)
    prev_scale_points_set: BoolProperty(default=False)

    # --- Flatten Bottom ---
    flatten_bottom_enabled: BoolProperty(
        name="Flatten Bottom",
        description=(
            "Boolean-cut the bottom of the model to create a perfectly flat "
            "base surface for 3D printing"
        ),
        default=False,
    )
    flatten_zone_height: FloatProperty(
        name="Zone Height (mm)",
        description=(
            "Defines the 'bottom zone': all vertices within this distance "
            "above the lowest point are considered part of the base.  "
            "The cut plane is placed at the TOP of this zone so that every "
            "surface in the zone becomes a single flat plane"
        ),
        default=0.2,
        min=0.001,
        soft_max=10.0,
        precision=3,
    )
    flatten_solver: EnumProperty(
        name="Solver",
        description="Boolean solver for the flatten cut",
        items=[
            ("EXACT", "Exact", "Slower but most accurate (recommended)"),
            ("MANIFOLD", "Manifold", "Good for complex geometry"),
            ("FLOAT", "Float", "Fast, works for simple shapes"),
        ],
        default="EXACT",
    )

    # --- Bottom Female Connector ---
    bottom_connector_enabled: BoolProperty(
        name="Bottom Female Connector",
        description=(
            "Cut an octagonal pocket (female connector) into the bottom of the model. "
            "6.3 mm wide, 6.4 mm deep, 0.4 mm fillet, 0.4 mm entry chamfer"
        ),
        default=False,
    )
    bottom_conn_offset_x: FloatProperty(
        name="X Offset (mm)",
        description=(
            "Shift the connector pocket left/right (X axis) from the "
            "auto-detected centre of gravity"
        ),
        default=0.0,
        soft_min=-10.0,
        soft_max=10.0,
        precision=2,
    )
    bottom_conn_offset_y: FloatProperty(
        name="Y Offset (mm)",
        description=(
            "Shift the connector pocket forward/backward (Y axis) from the "
            "auto-detected centre of gravity"
        ),
        default=0.0,
        soft_min=-10.0,
        soft_max=10.0,
        precision=2,
    )

    # --- Connectors for Features ---
    feature_connector_enabled: BoolProperty(
        name="Connectors for Features",
        description="Use BFS flood-fill from seed vertices to detect features and cut connectors",
        default=False,
    )
    gradient_threshold: FloatProperty(
        name="Max Gradient Angle (deg)",
        description="Override angle for feature selection. Leave at 0 to auto-detect when running",
        default=0.0,
        min=0.0,
        max=90.0,
        step=100,
        precision=1,
    )
    gradient_range_min: FloatProperty(
        name="Scan Min (deg)",
        description="Start of the angle range to scan for auto-detection",
        default=0.0,
        min=0.0,
        max=89.0,
        step=100,
        precision=0,
    )
    gradient_range_max: FloatProperty(
        name="Scan Max (deg)",
        description="End of the angle range to scan for auto-detection",
        default=30.0,
        min=2.0,
        max=90.0,
        step=100,
        precision=0,
    )
    feature_seeds_set: BoolProperty(default=False)
    feature_seed_count: IntProperty(default=0)

    # --- Connectors for Features: Previous (for restore) ---
    prev_feature_seeds_set: BoolProperty(default=False)
    prev_feature_seed_count: IntProperty(default=0)

    connector_depth: FloatProperty(
        name="Connector Depth",
        description="Depth behind the furthest point along the chosen direction (in scene units)",
        default=2.0,
        min=0.01,
        soft_max=50.0,
        precision=2,
    )
    connector_clearance: FloatProperty(
        name="Clearance",
        description="Extra offset for the slot for 3D printing tolerance (in scene units)",
        default=0.2,
        min=0.0,
        soft_max=2.0,
        precision=2,
    )
    connector_neg_dir_clearance: BoolProperty(
        name="Negative Direction Clearance",
        description=(
            "Also extend the cutter opposite the extrusion direction, "
            "continuing the draft angle if present"
        ),
        default=False,
    )
    connector_neg_dir_clearance_value: FloatProperty(
        name="Neg Dir Clearance",
        description="Amount to extend the cutter opposite the extrusion direction",
        default=0.2,
        min=0.0,
        soft_max=2.0,
        precision=2,
    )
    connector_draft_enabled: BoolProperty(
        name="Draft",
        description=(
            "Taper the back of the connector toward the area centroid of its "
            "flat back face, like a CAD draft. Side walls slope inward at the "
            "draft angle"
        ),
        default=False,
    )
    connector_draft_angle: FloatProperty(
        name="Draft Angle (deg)",
        description="Draft angle measured from the extrusion direction",
        default=30.0,
        min=0.0,
        max=45.0,
        step=100,
        precision=1,
    )
    connector_smooth_enabled: BoolProperty(
        name="Boundary Smoothing",
        description=(
            "Laplacian-smooth the feature boundary loop (in the plane "
            "perpendicular to the extrusion direction) before building the "
            "connector. Removes jagged tessellation edges and keeps the "
            "connector and slot cutter perfectly matched"
        ),
        default=False,
    )
    connector_smooth_iterations: IntProperty(
        name="Smoothing Iterations",
        description="Number of Laplacian smoothing passes on the boundary loop",
        default=8,
        min=1,
        max=50,
    )
    connector_straight_cut_enabled: BoolProperty(
        name="Straight Cut",
        description=(
            "Instead of an extruded plug, separate the feature with a single "
            "best-fit plane through its boundary loop. The plane minimises "
            "the sum of squared distances to the boundary verts and is "
            "oriented into the chosen direction's hemisphere. Produces two "
            "closed solids — the body and the feature — with matching flat "
            "interfaces. Overrides depth, clearance, draft and smoothing"
        ),
        default=False,
    )
    connector_straight_offset_clearance: FloatProperty(
        name="Offset Clearance",
        description=(
            "Straight cut only: radial clearance for the body-side cleanup "
            "pocket. The boundary loop is offset outward by this amount in "
            "the cut plane before the pocket is subtracted from the body"
        ),
        default=0.2,
        min=0.0,
        soft_max=2.0,
        precision=2,
    )
    connector_straight_depth_clearance: FloatProperty(
        name="Depth Clearance",
        description=(
            "Straight cut only: depth of the body-side cleanup pocket "
            "measured into the body from the cut plane. Pulls back any body "
            "geometry that leans into the plane so the mated foot sits flush"
        ),
        default=0.2,
        min=0.0,
        soft_max=2.0,
        precision=2,
    )
    connector_direction: EnumProperty(
        name="Direction",
        description="Direction to extrude the feature connector",
        items=[
            ("NEG_X", "-X", "Negative X direction"),
            ("POS_X", "+X", "Positive X direction"),
            ("NEG_Y", "-Y", "Negative Y direction"),
            ("POS_Y", "+Y", "Positive Y direction"),
            ("NEG_Z", "-Z", "Negative Z direction"),
            ("POS_Z", "+Z", "Positive Z direction"),
            ("NEG_X_NEG_Y", "-X/-Y", "45° diagonal: Negative X & Negative Y"),
            ("NEG_X_POS_Y", "-X/+Y", "45° diagonal: Negative X & Positive Y"),
            ("POS_X_NEG_Y", "+X/-Y", "45° diagonal: Positive X & Negative Y"),
            ("POS_X_POS_Y", "+X/+Y", "45° diagonal: Positive X & Positive Y"),
            ("NEG_X_NEG_Z", "-X/-Z", "45° diagonal: Negative X & Negative Z"),
            ("NEG_X_POS_Z", "-X/+Z", "45° diagonal: Negative X & Positive Z"),
            ("POS_X_NEG_Z", "+X/-Z", "45° diagonal: Positive X & Negative Z"),
            ("POS_X_POS_Z", "+X/+Z", "45° diagonal: Positive X & Positive Z"),
            ("NEG_Y_NEG_Z", "-Y/-Z", "45° diagonal: Negative Y & Negative Z"),
            ("NEG_Y_POS_Z", "-Y/+Z", "45° diagonal: Negative Y & Positive Z"),
            ("POS_Y_NEG_Z", "+Y/-Z", "45° diagonal: Positive Y & Negative Z"),
            ("POS_Y_POS_Z", "+Y/+Z", "45° diagonal: Positive Y & Positive Z"),
        ],
        default="NEG_X",
    )
    connector_solver: EnumProperty(
        name="Solver",
        description="Boolean solver to use",
        items=[
            ("MANIFOLD", "Manifold", "Good for complex geometry"),
            ("EXACT", "Exact", "Slower but more accurate"),
            ("FLOAT", "Float", "Fast, works for simple shapes"),
        ],
        default="MANIFOLD",
    )

    # --- Boolean (clearance subtract) ---
    boolean_enabled: BoolProperty(
        name="Boolean",
        description=(
            "Subtract a clearance-offset copy of the tool body from the target "
            "body. The offset grows in world +X and the YZ plane but never in "
            "-X, so the back face stays put. Both bodies are kept"
        ),
        default=False,
    )
    boolean_tool: bpy.props.PointerProperty(
        name="Tool",
        description="Body whose offset shape is subtracted from the target",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
    )
    boolean_target: bpy.props.PointerProperty(
        name="Target",
        description="Body that the offset tool shape is subtracted from",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
    )
    boolean_clearance: FloatProperty(
        name="Clearance (mm)",
        description="Outward offset applied to the tool before subtraction",
        default=0.2,
        min=0.0,
        soft_max=2.0,
        precision=3,
    )
    boolean_solver: EnumProperty(
        name="Solver",
        description="Boolean solver for the clearance subtraction",
        items=[
            ("EXACT", "Exact", "Slower but most accurate (recommended)"),
            ("MANIFOLD", "Manifold", "Good for complex geometry"),
            ("FLOAT", "Float", "Fast, works for simple shapes"),
        ],
        default="EXACT",
    )

    # --- Mark Left / Right ---
    mark_left_right_enabled: BoolProperty(
        name="Mark Left & Right",
        description=(
            "Engrave an L or r mark on the back (-X) face of the active "
            "object to identify left or right parts"
        ),
        default=False,
    )
    mark_offset_x: FloatProperty(
        name="X Offset (mm)",
        description=(
            "Shift the mark along the local X+ of the chosen face (toward "
            "the right side of the L, where its foot extends)"
        ),
        default=0.0,
        soft_min=-10.0,
        soft_max=10.0,
        precision=2,
    )
    mark_offset_y: FloatProperty(
        name="Y Offset (mm)",
        description=(
            "Shift the mark along the local Y+ of the chosen face (toward "
            "the top of the L)"
        ),
        default=0.0,
        soft_min=-10.0,
        soft_max=10.0,
        precision=2,
    )
    mark_direction: EnumProperty(
        name="Direction",
        description="Face direction to engrave the mark on",
        items=[
            ("NEG_X", "-X", "Negative X face (back)"),
            ("POS_X", "+X", "Positive X face (front)"),
            ("NEG_Y", "-Y", "Negative Y face"),
            ("POS_Y", "+Y", "Positive Y face"),
            ("NEG_Z", "-Z", "Negative Z face (bottom)"),
            ("POS_Z", "+Z", "Positive Z face (top)"),
            ("NEG_X_NEG_Y", "-X/-Y", "45° diagonal face: Negative X & Negative Y"),
            ("NEG_X_POS_Y", "-X/+Y", "45° diagonal face: Negative X & Positive Y"),
            ("POS_X_NEG_Y", "+X/-Y", "45° diagonal face: Positive X & Negative Y"),
            ("POS_X_POS_Y", "+X/+Y", "45° diagonal face: Positive X & Positive Y"),
            ("NEG_X_NEG_Z", "-X/-Z", "45° diagonal face: Negative X & Negative Z"),
            ("NEG_X_POS_Z", "-X/+Z", "45° diagonal face: Negative X & Positive Z"),
            ("POS_X_NEG_Z", "+X/-Z", "45° diagonal face: Positive X & Negative Z"),
            ("POS_X_POS_Z", "+X/+Z", "45° diagonal face: Positive X & Positive Z"),
            ("NEG_Y_NEG_Z", "-Y/-Z", "45° diagonal face: Negative Y & Negative Z"),
            ("NEG_Y_POS_Z", "-Y/+Z", "45° diagonal face: Negative Y & Positive Z"),
            ("POS_Y_NEG_Z", "+Y/-Z", "45° diagonal face: Positive Y & Negative Z"),
            ("POS_Y_POS_Z", "+Y/+Z", "45° diagonal face: Positive Y & Positive Z"),
        ],
        default="NEG_X",
    )
    mark_solver: EnumProperty(
        name="Solver",
        description="Boolean solver for the mark cut",
        items=[
            ("EXACT", "Exact", "Slower but most accurate (recommended)"),
            ("MANIFOLD", "Manifold", "Good for complex geometry"),
            ("FLOAT", "Float", "Fast, works for simple shapes"),
        ],
        default="MANIFOLD",
    )

    # --- Export ---
    export_enabled: BoolProperty(
        name="Export STLs",
        description="Export all visible objects as individual STL files",
        default=False,
    )
    export_hidden: BoolProperty(
        name="Export Hidden",
        description="Include hidden objects in the export",
        default=False,
    )
    export_forward_axis: EnumProperty(
        name="Forward",
        items=[
            ("X", "X", ""),
            ("Y", "Y", ""),
            ("Z", "Z", ""),
            ("NEGATIVE_X", "-X", ""),
            ("NEGATIVE_Y", "-Y", ""),
            ("NEGATIVE_Z", "-Z", ""),
        ],
        default="X",
    )
    export_up_axis: EnumProperty(
        name="Up",
        items=[
            ("X", "X", ""),
            ("Y", "Y", ""),
            ("Z", "Z", ""),
            ("NEGATIVE_X", "-X", ""),
            ("NEGATIVE_Y", "-Y", ""),
            ("NEGATIVE_Z", "-Z", ""),
        ],
        default="Z",
    )


# ===========================================================================
# Scale to Height - Set / Clear Reference Vertices
# ===========================================================================


class CPIPE_OT_set_scale_points(bpy.types.Operator):
    """Store the two selected vertices as the height reference"""

    bl_idname = "cpipe.set_scale_points"
    bl_label = "Set Scale Vertices"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == "MESH" and obj.mode == "EDIT"

    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()

        selected = [v for v in bm.verts if v.select]
        if len(selected) != 2:
            self.report(
                {"WARNING"}, f"Select exactly 2 vertices ({len(selected)} selected)"
            )
            return {"CANCELLED"}

        props = context.scene.cpipe

        world = obj.matrix_world
        p1 = world @ selected[0].co
        p2 = world @ selected[1].co

        props.point_a = p1
        props.point_b = p2
        props.ref_distance = abs(p1.z - p2.z)
        props.scale_points_set = True

        scale_len = context.scene.unit_settings.scale_length
        dist_mm = props.ref_distance * scale_len * 1000
        self.report({"INFO"}, f"Reference height (Z): {dist_mm:.2f} mm")
        return {"FINISHED"}


class CPIPE_OT_clear_scale_points(bpy.types.Operator):
    """Clear the stored reference vertices"""

    bl_idname = "cpipe.clear_scale_points"
    bl_label = "Clear Scale Vertices"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.cpipe
        props.scale_points_set = False
        props.ref_distance = 0.0
        self.report({"INFO"}, "Scale reference vertices cleared")
        return {"FINISHED"}


class CPIPE_OT_restore_scale_points(bpy.types.Operator):
    """Restore the previously stored scale reference vertices"""

    bl_idname = "cpipe.restore_scale_points"
    bl_label = "Restore Previous Vertices"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene.cpipe.prev_scale_points_set

    def execute(self, context):
        props = context.scene.cpipe
        props.point_a = props.prev_point_a.copy()
        props.point_b = props.prev_point_b.copy()
        props.ref_distance = props.prev_ref_distance
        props.scale_points_set = True

        scale_len = context.scene.unit_settings.scale_length
        dist_mm = props.ref_distance * scale_len * 1000
        self.report({"INFO"}, f"Restored scale vertices (Z height: {dist_mm:.2f} mm)")
        return {"FINISHED"}


# ===========================================================================
# Connectors for Features - Auto-detect / Set / Clear Feature Vertices
# ===========================================================================


class CPIPE_OT_set_feature_seeds(bpy.types.Operator):
    """Auto-detect all feature vertices from the current selection using BFS flood fill"""

    bl_idname = "cpipe.set_feature_seeds"
    bl_label = "Auto-detect Vertices"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == "MESH" and obj.mode == "EDIT"

    def execute(self, context):
        obj = context.active_object
        props = context.scene.cpipe
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.normal_update()

        seeds = [v for v in bm.verts if v.select]
        if not seeds:
            self.report(
                {"WARNING"}, "Select at least one seed vertex inside the feature"
            )
            return {"CANCELLED"}

        seed_indices = [v.index for v in seeds]

        if props.gradient_threshold < 0.5:
            optimal, _ = detect_optimal_angle(
                bm,
                seed_indices,
                props.gradient_range_min,
                props.gradient_range_max,
            )
            props.gradient_threshold = optimal
            self.report({"INFO"}, f"Auto-detected angle: {optimal:.0f} deg")

        grad_limit = math.radians(props.gradient_threshold)
        selected, boundary = bfs_feature_fill(bm, seed_indices, grad_limit)
        selected |= boundary

        bpy.ops.mesh.select_all(action="DESELECT")
        bm.verts.ensure_lookup_table()
        for idx in selected:
            bm.verts[idx].select = True
        bm.select_flush(True)
        bmesh.update_edit_mesh(obj.data)

        self.report(
            {"INFO"},
            f"Auto-detected {len(selected)} verts (boundary: {len(boundary)})",
        )
        return {"FINISHED"}


class CPIPE_OT_clear_feature_seeds(bpy.types.Operator):
    """Clear the stored feature vertices"""

    bl_idname = "cpipe.clear_feature_seeds"
    bl_label = "Clear Feature Vertices"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = context.active_object
        if obj and "cpipe_feature_seeds" in obj:
            del obj["cpipe_feature_seeds"]
        props = context.scene.cpipe
        props.feature_seeds_set = False
        props.feature_seed_count = 0
        props.gradient_threshold = 0.0
        self.report({"INFO"}, "Feature vertices cleared")
        return {"FINISHED"}


class CPIPE_OT_restore_feature_seeds(bpy.types.Operator):
    """Restore the previously stored feature vertices"""

    bl_idname = "cpipe.restore_feature_seeds"
    bl_label = "Restore Previous Vertices"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        props = context.scene.cpipe
        if not props.prev_feature_seeds_set:
            return False
        obj = context.active_object
        return obj is not None and "cpipe_prev_feature_seeds" in obj

    def execute(self, context):
        obj = context.active_object
        props = context.scene.cpipe

        prev_seeds = list(obj.get("cpipe_prev_feature_seeds", []))
        if not prev_seeds:
            self.report({"WARNING"}, "No previous vertices found")
            return {"CANCELLED"}

        obj["cpipe_feature_seeds"] = prev_seeds
        props.feature_seeds_set = True
        props.feature_seed_count = len(prev_seeds)
        props.gradient_threshold = 0.0

        self.report({"INFO"}, f"Restored {len(prev_seeds)} previous feature vertices")
        return {"FINISHED"}


# ===========================================================================
# Set Feature Vertices (store current selection)
# ===========================================================================


class CPIPE_OT_feature_select(bpy.types.Operator):
    """Store the currently selected vertices as feature vertices for the connector"""

    bl_idname = "cpipe.feature_select"
    bl_label = "Set Feature Vertices"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == "MESH" and obj.mode == "EDIT"

    def execute(self, context):
        obj = context.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()

        selected = [v for v in bm.verts if v.select]
        if not selected:
            self.report(
                {"WARNING"}, "Select at least one vertex to set as feature vertices"
            )
            return {"CANCELLED"}

        obj["cpipe_feature_seeds"] = [v.index for v in selected]

        props = context.scene.cpipe
        props.feature_seeds_set = True
        props.feature_seed_count = len(selected)

        self.report({"INFO"}, f"Stored {len(selected)} feature vertices")
        return {"FINISHED"}


# ===========================================================================
# Mark Left / Right
# ===========================================================================


class CPIPE_OT_mark_side(bpy.types.Operator):
    """Engrave an 'L' or 'r' mark on the back face to identify left or right.
    Auto-detects separate bodies: the left-most body (lowest Y centroid)
    gets the L mark, the right-most body gets the r mark."""

    bl_idname = "cpipe.mark_side"
    bl_label = "Mark Side"
    bl_options = {"REGISTER", "UNDO"}

    side: EnumProperty(
        name="Side",
        items=[
            ("LEFT", "Left", "Capital L mark (left part)"),
            ("RIGHT", "Right", "Capital T mark (right part)"),
        ],
        default="LEFT",
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == "MESH"

    def execute(self, context):
        obj = context.active_object
        props = context.scene.cpipe

        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        me = obj.data
        if len(me.vertices) == 0:
            self.report({"WARNING"}, "Mesh has no vertices")
            return {"CANCELLED"}

        # ---- Detect mesh islands ----
        islands = _find_mesh_islands(obj)

        if len(islands) < 2:
            # Single body — just mark its back face
            target_island = set(range(len(me.vertices)))
            self.report(
                {"INFO"},
                "Single body detected — marking its back face",
            )
        else:
            # Multiple bodies — sort by Y centroid to determine left/right.
            # In Blender's default front view (looking down -Y), the screen-
            # left body has a *lower* Y centroid.  We sort ascending by Y so
            # index 0 = left-most, index -1 = right-most.
            islands_sorted = sorted(
                islands,
                key=lambda isle: _island_centroid(obj, isle).y,
            )
            if self.side == "LEFT":
                target_island = islands_sorted[-1]
            else:
                target_island = islands_sorted[0]

            self.report(
                {"INFO"},
                f"Detected {len(islands)} bodies — targeting "
                f"{'left-most' if self.side == 'LEFT' else 'right-most'} "
                f"({len(target_island)} verts)",
            )

        # ---- Find face centre of the chosen island ----
        direction = props.mark_direction
        face_centre = _island_face_centre(obj, target_island, direction)

        # ---- Build the T-mark cutter ----
        mark_bm = create_tmark_cutter_bm(side=self.side)

        # Rotate the cutter from default orientation (profile on YZ,
        # extending +X) to match the chosen direction.
        rot = _mark_rotation_matrix(direction)
        bmesh.ops.transform(mark_bm, matrix=rot, verts=mark_bm.verts[:])

        mark_mesh = bpy.data.meshes.new("_TMarkCutter")
        mark_bm.to_mesh(mark_mesh)
        mark_bm.free()

        mark_obj = bpy.data.objects.new("_TMarkCutter", mark_mesh)
        context.collection.objects.link(mark_obj)

        # Offsets are expressed in the face's local frame, where X+ points
        # toward the right of the L (where its foot extends) and Y+ points
        # toward the top of the L.  The cutter profile is built on the
        # YZ plane with the L's foot extending toward -Y_profile and the
        # stem rising toward +Z_profile, so the local axes in world space
        # are just those two unit vectors rotated by the same matrix that
        # aligns the cutter with the chosen direction.
        rot3 = rot.to_3x3()
        local_x_dir = rot3 @ Vector((0, -1, 0))
        local_y_dir = rot3 @ Vector((0, 0, 1))
        mark_obj.location = face_centre + (
            local_x_dir * props.mark_offset_x + local_y_dir * props.mark_offset_y
        )
        context.view_layer.update()

        solver = props.mark_solver
        ok = apply_boolean_difference(context, obj, mark_obj, solver, self.report)

        if ok:
            label = "Left (L)" if self.side == "LEFT" else "Right (T)"
            self.report({"INFO"}, f"Marked as {label}")
        else:
            self.report(
                {"WARNING"},
                "Mark boolean failed. Try a different solver.",
            )
            return {"CANCELLED"}

        return {"FINISHED"}


# ===========================================================================
# Build Solid Helper (for Feature Connectors)
# ===========================================================================


def _parse_direction(direction):
    """Return a normalised direction Vector for a direction enum value."""
    s = 1.0 / math.sqrt(2)
    direction_map = {
        "NEG_X": Vector((-1, 0, 0)),
        "POS_X": Vector((1, 0, 0)),
        "NEG_Y": Vector((0, -1, 0)),
        "POS_Y": Vector((0, 1, 0)),
        "NEG_Z": Vector((0, 0, -1)),
        "POS_Z": Vector((0, 0, 1)),
        "NEG_X_NEG_Y": Vector((-s, -s, 0)),
        "NEG_X_POS_Y": Vector((-s, s, 0)),
        "POS_X_NEG_Y": Vector((s, -s, 0)),
        "POS_X_POS_Y": Vector((s, s, 0)),
        "NEG_X_NEG_Z": Vector((-s, 0, -s)),
        "NEG_X_POS_Z": Vector((-s, 0, s)),
        "POS_X_NEG_Z": Vector((s, 0, -s)),
        "POS_X_POS_Z": Vector((s, 0, s)),
        "NEG_Y_NEG_Z": Vector((0, -s, -s)),
        "NEG_Y_POS_Z": Vector((0, -s, s)),
        "POS_Y_NEG_Z": Vector((0, s, -s)),
        "POS_Y_POS_Z": Vector((0, s, s)),
    }
    return direction_map.get(direction, Vector((-1, 0, 0)))


def _planar_basis(dir_vec):
    """Return (u_axis, v_axis): an orthonormal basis for the plane perpendicular
    to *dir_vec*.
    """
    ref = Vector((0, 0, 1)) if abs(dir_vec.z) < 0.9 else Vector((1, 0, 0))
    u_axis = dir_vec.cross(ref).normalized()
    v_axis = dir_vec.cross(u_axis).normalized()
    return u_axis, v_axis


def build_boundary_loops(edge_face_count):
    """Given ``edge_face_count`` (keys: sorted ``(vi1, vi2)`` tuples, values:
    list of face indices touching that edge), walk every edge that belongs to
    exactly one face (a feature-boundary edge) and return ordered vertex loops.

    Each returned loop is a list of vertex indices in cyclic order.  Open
    chains (rare — would indicate non-manifold selection) are included as-is
    without wrapping.
    """
    adj = {}
    for (vi1, vi2), face_list in edge_face_count.items():
        if len(face_list) != 1:
            continue
        adj.setdefault(vi1, []).append(vi2)
        adj.setdefault(vi2, []).append(vi1)

    loops = []
    visited = set()
    for start in adj:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev = None
        cur = start
        while True:
            nxts = [n for n in adj[cur] if n != prev]
            if not nxts:
                break
            nxt = None
            for cand in nxts:
                if cand == start:
                    nxt = cand
                    break
                if cand not in visited:
                    nxt = cand
                    break
            if nxt is None or nxt == start:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev = cur
            cur = nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def smooth_boundary_loops(loops, vert_coords, dir_vec, iterations=8, factor=0.5):
    """Laplacian-relax boundary-loop vertices in the plane perpendicular to
    *dir_vec*.

    Only the in-plane component of each Laplacian step is applied — the axial
    component is preserved so verts stay on the feature surface.  Modifies
    *vert_coords* in place.  Interior feature verts are left untouched.
    """
    if iterations <= 0:
        return
    for _ in range(iterations):
        updates = {}
        for loop in loops:
            n = len(loop)
            if n < 3:
                continue
            for i, vi in enumerate(loop):
                prev_co = vert_coords[loop[(i - 1) % n]]
                next_co = vert_coords[loop[(i + 1) % n]]
                co = vert_coords[vi]
                delta = (prev_co + next_co) * 0.5 - co
                axial = dir_vec * dir_vec.dot(delta)
                in_plane = delta - axial
                updates[vi] = co + in_plane * factor
        for vi, new_co in updates.items():
            vert_coords[vi] = new_co


def boundary_2d_outward_normals(loops, vert_coords, dir_vec):
    """For each vertex of every boundary loop, compute the 2D outward unit
    normal in the plane perpendicular to *dir_vec*.

    The tangent at each vertex is (next − prev) projected into the plane; the
    normal is obtained by rotating the tangent 90° about *dir_vec*.  The sign
    is chosen so the normal points away from the loop's 2D centroid.

    Returns ``dict[int, Vector]`` keyed by vertex index.
    """
    u_axis, v_axis = _planar_basis(dir_vec)
    normals = {}
    for loop in loops:
        n = len(loop)
        if n < 3:
            continue
        # 2D centroid for outward disambiguation.
        cu = sum(u_axis.dot(vert_coords[vi]) for vi in loop) / n
        cv = sum(v_axis.dot(vert_coords[vi]) for vi in loop) / n
        for i, vi in enumerate(loop):
            prev_co = vert_coords[loop[(i - 1) % n]]
            next_co = vert_coords[loop[(i + 1) % n]]
            tang = next_co - prev_co
            tang = tang - dir_vec * dir_vec.dot(tang)
            nrm = dir_vec.cross(tang)
            if nrm.length < 1e-9:
                continue
            nrm.normalize()
            cu_off = u_axis.dot(vert_coords[vi]) - cu
            cv_off = v_axis.dot(vert_coords[vi]) - cv
            if u_axis.dot(nrm) * cu_off + v_axis.dot(nrm) * cv_off < 0:
                nrm = -nrm
            normals[vi] = nrm
    return normals


def build_solid_bmesh(
    face_vert_lists,
    vert_coords,
    selected_verts_set,
    edge_face_count,
    depth,
    clearance=0.0,
    back_trim=0.5,
    direction="NEG_X",
    draft_angle_rad=0.0,
    forward_clearance=0.0,
    boundary_2d_normals=None,
):
    dir_vec = _parse_direction(direction)

    # Find the extremum projection along the extrusion direction.
    projs = [dir_vec.dot(vert_coords[vi]) for vi in selected_verts_set]
    extremum_proj = max(projs)

    # Build deeper than requested, then bisect to get a clean back face.
    build_depth = depth + back_trim
    back_proj = extremum_proj + build_depth

    # Negative direction clearance: shift the front face opposite the extrusion
    # direction and, when drafted, widen it so the wall slope continues.
    forward_shift = (
        -dir_vec * forward_clearance if forward_clearance > 1e-6 else Vector((0, 0, 0))
    )
    forward_expand = (
        forward_clearance * math.tan(draft_angle_rad)
        if forward_clearance > 1e-6 and draft_angle_rad > 1e-6
        else 0.0
    )
    if forward_expand > 0.0:
        ref = Vector((0, 0, 1)) if abs(dir_vec.z) < 0.9 else Vector((1, 0, 0))
        fu_axis = dir_vec.cross(ref).normalized()
        fv_axis = dir_vec.cross(fu_axis).normalized()

        # Area-weighted centroid of the feature projected onto (fu, fv).
        total_area = 0.0
        wx = 0.0
        wy = 0.0
        for fvl in face_vert_lists:
            if len(fvl) < 3:
                continue
            pts = [
                (fu_axis.dot(vert_coords[vi]), fv_axis.dot(vert_coords[vi]))
                for vi in fvl
            ]
            p0x, p0y = pts[0]
            for i in range(1, len(pts) - 1):
                p1x, p1y = pts[i]
                p2x, p2y = pts[i + 1]
                tri_area = 0.5 * ((p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x))
                cx = (p0x + p1x + p2x) / 3.0
                cy = (p0y + p1y + p2y) / 3.0
                total_area += tri_area
                wx += tri_area * cx
                wy += tri_area * cy

        if abs(total_area) < 1e-12:
            forward_expand = 0.0
        else:
            fcu = wx / total_area
            fcv = wy / total_area

    bm = bmesh.new()
    front_map = {}
    back_map = {}

    for vi in selected_verts_set:
        co = vert_coords[vi]
        front_co = co + forward_shift
        if forward_expand > 0.0:
            vu = fu_axis.dot(co)
            vv = fv_axis.dot(co)
            du = vu - fcu
            dv = vv - fcv
            dist = math.sqrt(du * du + dv * dv)
            if dist > 1e-5:
                factor = forward_expand / dist
                front_co = front_co + fu_axis * (du * factor) + fv_axis * (dv * factor)
        front_v = bm.verts.new(front_co)
        front_map[vi] = front_v
        # Move co along dir_vec until its projection equals back_proj.
        offset = back_proj - dir_vec.dot(co)
        back_v = bm.verts.new(co + dir_vec * offset)
        back_map[vi] = back_v

    bm.verts.ensure_lookup_table()
    bm.verts.index_update()

    for fvl in face_vert_lists:
        fverts = [front_map[vi] for vi in fvl]
        try:
            bm.faces.new(fverts)
        except ValueError:
            pass

    for fvl in face_vert_lists:
        bverts = [back_map[vi] for vi in fvl]
        bverts.reverse()
        try:
            bm.faces.new(bverts)
        except ValueError:
            pass

    for (vi1, vi2), face_list in edge_face_count.items():
        if len(face_list) != 1:
            continue
        fv1, fv2 = front_map[vi1], front_map[vi2]
        bv1, bv2 = back_map[vi1], back_map[vi2]
        adj_fvl = None
        for fvl in face_vert_lists:
            if vi1 in fvl and vi2 in fvl:
                adj_fvl = fvl
                break
        if adj_fvl is None:
            continue
        idx1 = adj_fvl.index(vi1)
        idx2 = adj_fvl.index(vi2)
        if (idx1 + 1) % len(adj_fvl) == idx2:
            quad = [fv1, bv1, bv2, fv2]
        else:
            quad = [fv2, bv2, bv1, fv1]
        try:
            bm.faces.new(quad)
        except ValueError:
            try:
                quad.reverse()
                bm.faces.new(quad)
            except ValueError:
                pass

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    # Bisect at the intended back depth to trim thin geometry and create
    # a clean, flat back face.
    cut_proj = extremum_proj + depth
    plane_co = dir_vec * cut_proj
    plane_no = -dir_vec  # points inward (toward the feature)
    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
    result = bmesh.ops.bisect_plane(
        bm,
        geom=geom,
        plane_co=plane_co,
        plane_no=plane_no,
        clear_outer=False,
        clear_inner=True,
    )

    # Fill the open cut boundary to close the back.
    cut_edges = [e for e in result["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
    if cut_edges:
        bmesh.ops.contextual_create(bm, geom=cut_edges)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    # Draft: taper the flat back inward toward its area centroid. Applied
    # before clearance so the cutter inflates a drafted connector uniformly.
    if draft_angle_rad > 1e-6 and depth > 1e-6:
        shrink = depth * math.tan(draft_angle_rad)

        # Orthonormal basis (u, v) perpendicular to dir_vec.
        ref = Vector((0, 0, 1)) if abs(dir_vec.z) < 0.9 else Vector((1, 0, 0))
        u_axis = dir_vec.cross(ref).normalized()
        v_axis = dir_vec.cross(u_axis).normalized()

        bm.faces.ensure_lookup_table()
        plane_eps = 1e-4
        back_faces = [
            f
            for f in bm.faces
            if all(abs(dir_vec.dot(fv.co) - cut_proj) < plane_eps for fv in f.verts)
        ]

        for back_face in back_faces:
            face_verts = list(back_face.verts)
            if len(face_verts) < 3:
                continue

            pts_2d = [(u_axis.dot(fv.co), v_axis.dot(fv.co)) for fv in face_verts]

            # Area centroid via fan triangulation from pts_2d[0].
            p0x, p0y = pts_2d[0]
            total_area = 0.0
            wx = 0.0
            wy = 0.0
            for i in range(1, len(pts_2d) - 1):
                p1x, p1y = pts_2d[i]
                p2x, p2y = pts_2d[i + 1]
                tri_area = 0.5 * ((p1x - p0x) * (p2y - p0y) - (p1y - p0y) * (p2x - p0x))
                cx = (p0x + p1x + p2x) / 3.0
                cy = (p0y + p1y + p2y) / 3.0
                total_area += tri_area
                wx += tri_area * cx
                wy += tri_area * cy

            if abs(total_area) < 1e-12:
                continue

            cu = wx / total_area
            cv = wy / total_area

            eps = 1e-5
            for fv, (vu, vv) in zip(face_verts, pts_2d):
                du = cu - vu
                dv = cv - vv
                dist = math.sqrt(du * du + dv * dv)
                if dist < eps:
                    continue
                move = min(shrink, dist - eps)
                factor = move / dist
                fv.co += u_axis * (du * factor) + v_axis * (dv * factor)

        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    if clearance > 0.0:
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        # Compute 2D outward normals on the fly if caller didn't supply them.
        if boundary_2d_normals is None:
            loops = build_boundary_loops(edge_face_count)
            boundary_2d_normals = boundary_2d_outward_normals(
                loops, vert_coords, dir_vec
            )

        # Front-cap verts survive bisect; back-cap verts at cut_proj were
        # created by bisect + contextual_create.
        front_vert_to_vi = {fv: vi for vi, fv in front_map.items()}
        proj_eps = 1e-4
        back_vert_set = set()
        for v in bm.verts:
            if v in front_vert_to_vi:
                continue
            if abs(dir_vec.dot(v.co) - cut_proj) < proj_eps:
                back_vert_set.add(v)

        u_axis, v_axis = _planar_basis(dir_vec)

        # 2D centroid of the back cap, used to orient outward normals along
        # the back loop even after draft has moved verts inward.
        if back_vert_set:
            back_cu = sum(u_axis.dot(v.co) for v in back_vert_set) / len(back_vert_set)
            back_cv = sum(v_axis.dot(v.co) for v in back_vert_set) / len(back_vert_set)
        else:
            back_cu = back_cv = 0.0

        # Adjacency along the back loop (only edges that stay on cut_proj).
        back_adj = {v: [] for v in back_vert_set}
        for v in back_vert_set:
            for e in v.link_edges:
                other = e.other_vert(v)
                if other in back_vert_set:
                    back_adj[v].append(other)

        new_positions = {}

        # Front-cap verts: push forward (−dir_vec) so the cutter pokes out of
        # the model surface for a clean boolean.  Perimeter verts also move
        # outward in the plane by the clean 2D normal.
        for vi, fv in front_map.items():
            delta = -dir_vec * clearance
            n2d = boundary_2d_normals.get(vi)
            if n2d is not None:
                delta = delta + n2d * clearance
            new_positions[fv] = fv.co + delta

        # Back-cap verts: deeper axially and outward planar (all are on the
        # perimeter).  Planar direction is derived from two loop neighbours
        # on the back cap, disambiguated against the back-cap centroid.
        for v in back_vert_set:
            delta = dir_vec * clearance
            nbrs = back_adj.get(v, [])
            if len(nbrs) >= 2:
                tang = nbrs[1].co - nbrs[0].co
                tang = tang - dir_vec * dir_vec.dot(tang)
                nrm3d = dir_vec.cross(tang)
                if nrm3d.length > 1e-9:
                    nrm3d.normalize()
                    off_u = u_axis.dot(v.co) - back_cu
                    off_v = v_axis.dot(v.co) - back_cv
                    if u_axis.dot(nrm3d) * off_u + v_axis.dot(nrm3d) * off_v < 0:
                        nrm3d = -nrm3d
                    delta = delta + nrm3d * clearance
            new_positions[v] = v.co + delta

        for v, new_co in new_positions.items():
            v.co = new_co

    return bm


# ===========================================================================
# Straight Cut — Best-Fit Plane Bisection
# ===========================================================================


def fit_best_plane(points, preferred_normal=None):
    """Least-squares best-fit plane through *points* (world space).

    The plane passes through the centroid and has normal equal to the
    eigenvector of the smallest eigenvalue of the point-cloud covariance
    matrix — the direction of least variance, which minimises the sum of
    squared orthogonal distances from the points to the plane.

    When *preferred_normal* is supplied, the returned normal is flipped
    as needed so it lies in the same hemisphere (positive dot product).
    """
    n = len(points)
    fallback = (
        preferred_normal.normalized() if preferred_normal else Vector((0, 0, 1))
    )
    if n == 0:
        return Vector((0, 0, 0)), fallback

    centroid = Vector((0, 0, 0))
    for p in points:
        centroid = centroid + p
    centroid = centroid / n

    if n < 3:
        return centroid, fallback

    xx = xy = xz = yy = yz = zz = 0.0
    for p in points:
        d = p - centroid
        xx += d.x * d.x
        xy += d.x * d.y
        xz += d.x * d.z
        yy += d.y * d.y
        yz += d.y * d.z
        zz += d.z * d.z

    import numpy as _np
    cov = _np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    _, eigvecs = _np.linalg.eigh(cov)
    nrm = Vector(
        (float(eigvecs[0, 0]), float(eigvecs[1, 0]), float(eigvecs[2, 0]))
    )

    if nrm.length < 1e-9:
        nrm = fallback
    else:
        nrm.normalize()

    if preferred_normal is not None and nrm.dot(preferred_normal) < 0:
        nrm = -nrm

    return centroid, nrm


def split_by_feature_faces(
    obj, feature_face_set, boundary_loops, keep_feature, cap_loops=True
):
    """Build a closed bmesh from *obj.data* containing either the feature
    half (*keep_feature=True*) or the rest (*keep_feature=False*).

    Uses the existing face selection as the cut — not an infinite plane —
    so the split is confined to the boundary loop and never leaks into
    unrelated parts of the mesh.

    When *cap_loops* is True, each boundary loop is closed with one cap
    face.  If the caller has already snapped the boundary verts onto a
    common plane, the foot's cap and the body's cap coincide exactly and
    both halves share a matching flat interface.

    Callers that want to follow up with an infinite-plane bisect should
    pass ``cap_loops=False``; a pre-existing cap would overlap with the
    new edges the bisect creates on the plane.
    """
    me = obj.data
    bm = bmesh.new()

    # Mirror every vert so local indices match obj.data.vertices indices
    # (face and loop lookups below use those indices directly).
    for v in me.vertices:
        bm.verts.new(v.co)
    bm.verts.ensure_lookup_table()

    for fi, poly in enumerate(me.polygons):
        is_feature = fi in feature_face_set
        if keep_feature != is_feature:
            continue
        try:
            bm.faces.new([bm.verts[vi] for vi in poly.vertices])
        except ValueError:
            pass

    # Cap each boundary loop. The two halves need opposite winding so
    # each cap's outward normal points away from its own volume; we
    # flip for the body side and let recalc_face_normals sort any drift.
    if cap_loops:
        for loop in boundary_loops:
            cap_verts = [bm.verts[vi] for vi in loop]
            if not keep_feature:
                cap_verts = list(reversed(cap_verts))
            try:
                bm.faces.new(cap_verts)
            except ValueError:
                pass

    orphans = [v for v in bm.verts if not v.link_faces]
    if orphans:
        bmesh.ops.delete(bm, geom=orphans, context="VERTS")

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def build_straight_cut_body_cleanup_bm(
    loop_positions, plane_no_into_body, offset_clearance, depth_clearance
):
    """Build a prism cutter to carve a clearance pocket into the body
    along a straight-cut plane.

    The prism's front face sits on the cut plane — the loop offset
    outward (in the plane) by *offset_clearance* — and it extrudes
    *depth_clearance* into the body along *plane_no_into_body*.

    Boolean-subtracting this from the body gives the mated foot XY slop
    and pulls back any body geometry that leans into the cut plane,
    so the interface is genuinely planar on the body side too.

    ``loop_positions``: list of loops, each a list of Vectors on the
    cut plane (local space, already snapped).
    ``plane_no_into_body``: unit Vector pointing from the plane into
    the body (opposite of the foot side).
    """
    bm = bmesh.new()

    # Orthonormal basis (u, v) in the cut plane, used to resolve the
    # 2D outward direction for each loop vertex.
    ref = (
        Vector((0, 0, 1))
        if abs(plane_no_into_body.z) < 0.9
        else Vector((1, 0, 0))
    )
    u_axis = plane_no_into_body.cross(ref).normalized()
    v_axis = plane_no_into_body.cross(u_axis).normalized()

    for loop in loop_positions:
        n = len(loop)
        if n < 3:
            continue

        cu = sum(u_axis.dot(p) for p in loop) / n
        cv = sum(v_axis.dot(p) for p in loop) / n

        outward = []
        for i in range(n):
            prev_p = loop[(i - 1) % n]
            next_p = loop[(i + 1) % n]
            tang = next_p - prev_p
            tang = tang - plane_no_into_body * plane_no_into_body.dot(tang)
            nrm = plane_no_into_body.cross(tang)
            if nrm.length < 1e-9:
                outward.append(Vector((0, 0, 0)))
                continue
            nrm.normalize()
            cu_off = u_axis.dot(loop[i]) - cu
            cv_off = v_axis.dot(loop[i]) - cv
            if u_axis.dot(nrm) * cu_off + v_axis.dot(nrm) * cv_off < 0:
                nrm = -nrm
            outward.append(nrm)

        front_verts = []
        back_verts = []
        for i, p in enumerate(loop):
            front_co = p + outward[i] * offset_clearance
            back_co = front_co + plane_no_into_body * depth_clearance
            front_verts.append(bm.verts.new(front_co))
            back_verts.append(bm.verts.new(back_co))

        try:
            bm.faces.new(front_verts)
        except ValueError:
            pass
        try:
            bm.faces.new(list(reversed(back_verts)))
        except ValueError:
            pass
        for i in range(n):
            j = (i + 1) % n
            try:
                bm.faces.new(
                    [front_verts[i], back_verts[i], back_verts[j], front_verts[j]]
                )
            except ValueError:
                pass

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


# ===========================================================================
# YZ-Plane Offset Cutter (Boolean step)
# ===========================================================================


def build_yz_offset_cutter_bm(tool_obj, clearance):
    """Build a bmesh that is *tool_obj* baked into world space, then expanded
    outward by *clearance* along each vertex's averaged face normal — but with
    the normal's X component clamped to be non-negative first.

    The clamp means faces pointing in +X grow in +X, faces pointing in YZ grow
    in YZ, and faces pointing in -X don't move at all.  In practice the back
    (-X) face of the tool stays put while every other surface inflates.

    Returned bmesh is in world coordinates; the cutter object should sit at
    the origin (identity matrix_world) when used for the boolean.
    """
    bm = bmesh.new()
    bm.from_mesh(tool_obj.data)

    world = tool_obj.matrix_world
    for v in bm.verts:
        v.co = world @ v.co

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    if clearance > 0.0:
        for f in bm.faces:
            f.normal_update()
        for v in bm.verts:
            if not v.link_faces:
                continue
            avg_normal = Vector((0, 0, 0))
            for f in v.link_faces:
                avg_normal += f.normal
            if avg_normal.x < 0.0:
                avg_normal.x = 0.0
            if avg_normal.length > 0:
                avg_normal.normalize()
                v.co += avg_normal * clearance

    return bm


# ===========================================================================
# Centre of Gravity (Volume Centroid)
# ===========================================================================


def _mesh_center_of_gravity_xy(obj):
    """Compute the volumetric centroid of a mesh and return its (X, Y).

    Uses the signed-tetrahedron method: for each triangle face, form a
    tetrahedron with the origin.  The signed volume and weighted centroid
    of all tetrahedra give the true centre of mass (assuming uniform
    density).  This works correctly for any closed mesh regardless of
    topology.

    Falls back to bounding-box centre if the mesh has no faces or the
    total volume is degenerate (e.g. a flat plane).
    """
    me = obj.data
    world = obj.matrix_world

    # Ensure we have loop_triangles
    me.calc_loop_triangles()
    tris = me.loop_triangles

    if not tris:
        # Fallback: bbox centre
        corners = [world @ Vector(c) for c in obj.bound_box]
        cx = sum(c.x for c in corners) / 8.0
        cy = sum(c.y for c in corners) / 8.0
        return cx, cy

    total_vol = 0.0
    weighted_x = 0.0
    weighted_y = 0.0

    verts = me.vertices
    for tri in tris:
        v0 = world @ verts[tri.vertices[0]].co
        v1 = world @ verts[tri.vertices[1]].co
        v2 = world @ verts[tri.vertices[2]].co

        # Signed volume of tetrahedron formed with origin
        cross = v1.cross(v2)
        vol = v0.dot(cross) / 6.0

        # Centroid of tetrahedron = (v0 + v1 + v2 + origin) / 4
        #                         = (v0 + v1 + v2) / 4
        cx_t = (v0.x + v1.x + v2.x) / 4.0
        cy_t = (v0.y + v1.y + v2.y) / 4.0

        total_vol += vol
        weighted_x += vol * cx_t
        weighted_y += vol * cy_t

    if abs(total_vol) < 1e-12:
        # Degenerate — fall back to bbox centre
        corners = [world @ Vector(c) for c in obj.bound_box]
        cx = sum(c.x for c in corners) / 8.0
        cy = sum(c.y for c in corners) / 8.0
        return cx, cy

    cx = weighted_x / total_vol
    cy = weighted_y / total_vol
    return cx, cy


# ===========================================================================
# Shared Boolean Helper
# ===========================================================================


def apply_boolean_difference(context, target_obj, cutter_obj, solver, report_fn=None):
    """Apply a Boolean DIFFERENCE modifier and clean up the cutter.

    Returns True on success, False on failure.
    """
    bpy.ops.object.select_all(action="DESELECT")
    target_obj.select_set(True)
    context.view_layer.objects.active = target_obj

    bool_mod = target_obj.modifiers.new(name="_BoolCut", type="BOOLEAN")
    bool_mod.operation = "DIFFERENCE"
    bool_mod.object = cutter_obj
    bool_mod.solver = solver
    try:
        bool_mod.use_hole_tolerant = True
    except AttributeError:
        pass

    success = True
    try:
        bpy.ops.object.modifier_apply(modifier=bool_mod.name)
    except RuntimeError as e:
        if report_fn:
            report_fn({"WARNING"}, f"Boolean issue: {e}")
        success = False

    # Clean up cutter
    cutter_data = cutter_obj.data
    bpy.data.objects.remove(cutter_obj, do_unlink=True)
    bpy.data.meshes.remove(cutter_data)

    return success


# ===========================================================================
# Boolean (clearance subtract)
# ===========================================================================


def perform_boolean_subtract(
    context, tool_obj, target_obj, clearance, solver, report_fn=None
):
    """Subtract a YZ-plane offset of *tool_obj* from *target_obj*.

    Returns True on success, False on failure.  Both input objects are kept;
    only the temporary cutter mesh is removed.
    """
    if target_obj.mode != "OBJECT":
        bpy.ops.object.select_all(action="DESELECT")
        target_obj.select_set(True)
        context.view_layer.objects.active = target_obj
        bpy.ops.object.mode_set(mode="OBJECT")

    cutter_bm = build_yz_offset_cutter_bm(tool_obj, clearance)
    cutter_mesh = bpy.data.meshes.new("_BooleanCutter")
    cutter_bm.to_mesh(cutter_mesh)
    cutter_bm.free()

    cutter_obj = bpy.data.objects.new("_BooleanCutter", cutter_mesh)
    context.collection.objects.link(cutter_obj)
    context.view_layer.update()

    return apply_boolean_difference(context, target_obj, cutter_obj, solver, report_fn)


class CPIPE_OT_boolean(bpy.types.Operator):
    """Subtract a clearance-offset copy of the tool body from the target body"""

    bl_idname = "cpipe.boolean"
    bl_label = "Run Boolean"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        props = context.scene.cpipe
        return (
            props.boolean_tool is not None
            and props.boolean_target is not None
            and props.boolean_tool != props.boolean_target
        )

    def execute(self, context):
        props = context.scene.cpipe
        tool = props.boolean_tool
        target = props.boolean_target

        if tool is None or target is None:
            self.report({"WARNING"}, "Pick both a tool and a target body")
            return {"CANCELLED"}
        if tool == target:
            self.report({"WARNING"}, "Tool and target must be different objects")
            return {"CANCELLED"}
        if tool.type != "MESH" or target.type != "MESH":
            self.report({"WARNING"}, "Tool and target must be mesh objects")
            return {"CANCELLED"}

        ok = perform_boolean_subtract(
            context,
            tool,
            target,
            props.boolean_clearance,
            props.boolean_solver,
            self.report,
        )
        if not ok:
            self.report({"WARNING"}, "Boolean failed. Try a different solver.")
            return {"CANCELLED"}

        self.report(
            {"INFO"},
            f"Subtracted {tool.name} (offset {props.boolean_clearance:.3f} mm) from {target.name}",
        )
        return {"FINISHED"}


# ===========================================================================
# Run Pipeline
# ===========================================================================


class CPIPE_OT_run_pipeline(bpy.types.Operator):
    """Run the pipeline: Scale -> Flatten Bottom -> Bottom Connector -> Feature Connectors"""

    bl_idname = "cpipe.run_pipeline"
    bl_label = "Run Pipeline"
    bl_options = {"REGISTER", "UNDO"}

    directory: bpy.props.StringProperty(subtype="DIR_PATH")

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not (obj and obj.type == "MESH"):
            return False
        props = context.scene.cpipe
        return (
            (props.scale_enabled and props.scale_points_set)
            or props.flatten_bottom_enabled
            or props.bottom_connector_enabled
            or (props.feature_connector_enabled and props.feature_seeds_set)
            or (
                props.boolean_enabled
                and props.boolean_tool is not None
                and props.boolean_target is not None
                and props.boolean_tool != props.boolean_target
            )
            or props.export_enabled
        )

    def invoke(self, context, event):
        props = context.scene.cpipe
        if props.export_enabled:
            # Default to the blend file's directory
            if bpy.data.filepath:
                self.directory = os.path.dirname(bpy.data.filepath)
            context.window_manager.fileselect_add(self)
            return {"RUNNING_MODAL"}
        return self.execute(context)

    def execute(self, context):
        obj = context.active_object
        props = context.scene.cpipe
        did_scale = False
        did_flatten = False
        did_bottom_conn = False
        did_connector = False
        did_boolean = False
        did_export = False
        export_count = 0
        flatten_info = ""

        original_mode = obj.mode
        original_mesh_select_mode = tuple(context.tool_settings.mesh_select_mode)

        # ================================================================
        # STEP 1 — SCALE TO HEIGHT
        # ================================================================

        if props.scale_enabled and props.scale_points_set:
            if props.ref_distance < 1e-8:
                self.report({"WARNING"}, "Reference distance is zero - skipping scale")
            else:
                if obj.mode == "EDIT":
                    bpy.ops.object.mode_set(mode="OBJECT")

                old_unit_scale = context.scene.unit_settings.scale_length
                new_unit_scale = 0.001

                context.scene.unit_settings.system = "METRIC"
                context.scene.unit_settings.scale_length = new_unit_scale
                context.scene.unit_settings.length_unit = "MILLIMETERS"

                unit_compensation = old_unit_scale / new_unit_scale
                ref_distance_mm = props.ref_distance * unit_compensation

                scale_factor = props.target_height_mm / ref_distance_mm
                total_scale = unit_compensation * scale_factor
                obj.scale *= total_scale

                obj.select_set(True)
                context.view_layer.objects.active = obj
                bpy.ops.object.transform_apply(
                    location=False, rotation=False, scale=True
                )

                props.ref_distance = props.target_height_mm
                did_scale = True

                # Save current scale points as previous, then reset
                props.prev_point_a = props.point_a.copy()
                props.prev_point_b = props.point_b.copy()
                props.prev_ref_distance = props.ref_distance
                props.prev_scale_points_set = True

                props.scale_points_set = False
                props.ref_distance = 0.0

        # ================================================================
        # STEP 2 — FLATTEN BOTTOM  (Boolean cut)
        # ================================================================

        if props.flatten_bottom_enabled:
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")

            me = obj.data
            if len(me.vertices) == 0:
                self.report({"WARNING"}, "Mesh has no vertices - skipping flatten")
            else:
                cut_z, min_z, n_bottom = detect_bottom_cut_level(
                    obj, props.flatten_zone_height
                )

                removed_mm = cut_z - min_z

                if removed_mm < 1e-5:
                    self.report(
                        {"INFO"},
                        "Bottom is already flat within zone - nothing to cut",
                    )
                    did_flatten = True
                    flatten_info = "already flat"
                else:
                    # Build cutter
                    cutter = create_flatten_cutter_obj(context, obj, cut_z)
                    context.view_layer.update()

                    ok = apply_boolean_difference(
                        context, obj, cutter, props.flatten_solver, self.report
                    )
                    if ok:
                        did_flatten = True
                        flatten_info = (
                            f"cut {removed_mm:.3f} mm " f"({n_bottom} verts in zone)"
                        )
                    else:
                        self.report(
                            {"WARNING"},
                            "Flatten boolean failed. Try a different solver.",
                        )

        # ================================================================
        # STEP 3 — BOTTOM FEMALE CONNECTOR
        # ================================================================

        if props.bottom_connector_enabled:
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")

            me = obj.data
            verts_world = [obj.matrix_world @ v.co for v in me.vertices]

            if not verts_world:
                self.report(
                    {"WARNING"}, "Mesh has no vertices - skipping bottom connector"
                )
            else:
                min_z = min(v.z for v in verts_world)

                # Use center of gravity (volume centroid) for X/Y placement.
                # Approximate via signed tetrahedron volume method over the mesh
                # triangles.  Falls back to bbox centre if the mesh is non-manifold
                # or degenerate.
                cx, cy = _mesh_center_of_gravity_xy(obj)

                overlap = 0.01
                cutter_bm = create_octagon_cutter_bm(
                    depth=_BOTTOM_CONN_DEPTH + overlap,
                )

                cutter_mesh = bpy.data.meshes.new("_BottomConnCutter")
                cutter_bm.to_mesh(cutter_mesh)
                cutter_bm.free()

                cutter_obj = bpy.data.objects.new("_BottomConnCutter", cutter_mesh)
                context.collection.objects.link(cutter_obj)

                # Position: bottom of cutter slightly below model bottom,
                # pocket extends upward into the model.
                cutter_obj.location = Vector(
                    (
                        cx + props.bottom_conn_offset_x,
                        cy + props.bottom_conn_offset_y,
                        min_z - overlap,
                    )
                )
                context.view_layer.update()

                ok = apply_boolean_difference(
                    context, obj, cutter_obj, "EXACT", self.report
                )
                did_bottom_conn = ok

        # ================================================================
        # STEP 4 — CONNECTORS FOR FEATURES
        # ================================================================

        if props.feature_connector_enabled and props.feature_seeds_set:
            bpy.ops.object.mode_set(mode="EDIT")

            feature_vert_indices = list(obj.get("cpipe_feature_seeds", []))
            if not feature_vert_indices:
                self.report(
                    {"WARNING"}, "No feature vertices stored - skipping connector"
                )
            else:
                bm = bmesh.from_edit_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                bm.normal_update()

                feature_verts = set(feature_vert_indices)

                bpy.ops.mesh.select_all(action="DESELECT")
                bm.verts.ensure_lookup_table()
                bpy.ops.mesh.select_mode(type="FACE")
                for idx in feature_verts:
                    if idx < len(bm.verts):
                        bm.verts[idx].select = True
                bm.select_flush(True)
                bmesh.update_edit_mesh(obj.data)

                bm = bmesh.from_edit_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                bm.faces.ensure_lookup_table()

                selected_face_indices = [f.index for f in bm.faces if f.select]

                if not selected_face_indices:
                    self.report(
                        {"WARNING"},
                        "Feature select produced no faces. Try a different angle.",
                    )
                    context.tool_settings.mesh_select_mode = original_mesh_select_mode
                    return {"CANCELLED"}

                selected_faces_set = set(selected_face_indices)
                selected_verts_set = set()
                for fi in selected_faces_set:
                    for v in bm.faces[fi].verts:
                        selected_verts_set.add(v.index)

                face_vert_lists = []
                for fi in selected_faces_set:
                    face = bm.faces[fi]
                    face_vert_lists.append([v.index for v in face.verts])

                edge_face_count = {}
                for fi in selected_faces_set:
                    face = bm.faces[fi]
                    for edge in face.edges:
                        key = tuple(sorted([edge.verts[0].index, edge.verts[1].index]))
                        if key not in edge_face_count:
                            edge_face_count[key] = []
                        edge_face_count[key].append(fi)

                vert_coords = {}
                for vi in selected_verts_set:
                    vert_coords[vi] = bm.verts[vi].co.copy()

                depth = props.connector_depth
                clearance = props.connector_clearance

                bpy.ops.object.mode_set(mode="OBJECT")

                direction = props.connector_direction
                draft_rad = (
                    math.radians(props.connector_draft_angle)
                    if props.connector_draft_enabled
                    else 0.0
                )

                dir_vec = _parse_direction(direction)
                boundary_loops = build_boundary_loops(edge_face_count)

                if props.connector_straight_cut_enabled:
                    world_mat = obj.matrix_world
                    world_inv = world_mat.inverted()
                    boundary_vis = []
                    boundary_pts_world = []
                    seen = set()
                    for loop in boundary_loops:
                        for vi in loop:
                            if vi in seen:
                                continue
                            seen.add(vi)
                            boundary_vis.append(vi)
                            boundary_pts_world.append(
                                world_mat @ obj.data.vertices[vi].co
                            )

                    if len(boundary_pts_world) < 3:
                        self.report(
                            {"WARNING"},
                            "Boundary loop too small for straight cut",
                        )
                    else:
                        plane_co, plane_normal = fit_best_plane(
                            boundary_pts_world, preferred_normal=dir_vec
                        )

                        # Snap every boundary vert onto the best-fit plane
                        # so the cap face on each half is flat and the
                        # body's cap and the foot's cap coincide exactly.
                        for vi, pt in zip(boundary_vis, boundary_pts_world):
                            signed_dist = (pt - plane_co).dot(plane_normal)
                            snapped = pt - plane_normal * signed_dist
                            obj.data.vertices[vi].co = world_inv @ snapped
                        obj.data.update()

                        # Capture boundary loop positions in local space
                        # now, while indices still map into obj.data.
                        # body_bm.to_mesh(obj.data) below re-keys verts.
                        boundary_loops_local = [
                            [obj.data.vertices[vi].co.copy() for vi in loop]
                            for loop in boundary_loops
                        ]

                        foot_bm = split_by_feature_faces(
                            obj,
                            selected_faces_set,
                            boundary_loops,
                            keep_feature=True,
                            cap_loops=False,
                        )

                        # Infinite-plane cleanup: the feature faces can
                        # contain interior verts that sit *past* the
                        # best-fit plane (the small part's would-be flat
                        # bottom). Those overhangs stop the part from
                        # sitting flat on a printer plate. Cut the foot
                        # with the same plane and throw away whichever
                        # side has fewer verts — that's the thin sliver
                        # of overhang; the bulk of the foot survives.
                        # Picking the keep-side from geometry rather
                        # than dir_vec handles both connector directions
                        # (the user can pick either, and only the sign
                        # of the preferred normal flips with them).
                        local_plane_co = world_inv @ plane_co
                        local_plane_no = (
                            world_mat.to_3x3().transposed() @ plane_normal
                        ).normalized()

                        pos = neg = 0
                        for v in foot_bm.verts:
                            s = local_plane_no.dot(v.co - local_plane_co)
                            if s > 1e-6:
                                pos += 1
                            elif s < -1e-6:
                                neg += 1
                        if neg > pos:
                            local_plane_no = -local_plane_no

                        geom = (
                            foot_bm.verts[:]
                            + foot_bm.edges[:]
                            + foot_bm.faces[:]
                        )
                        bmesh.ops.bisect_plane(
                            foot_bm,
                            geom=geom,
                            plane_co=local_plane_co,
                            plane_no=local_plane_no,
                            clear_outer=False,
                            clear_inner=True,
                        )

                        open_edges = [
                            e for e in foot_bm.edges if len(e.link_faces) == 1
                        ]
                        if open_edges:
                            bmesh.ops.contextual_create(
                                foot_bm, geom=open_edges
                            )
                        foot_orphans = [
                            v for v in foot_bm.verts if not v.link_faces
                        ]
                        if foot_orphans:
                            bmesh.ops.delete(
                                foot_bm, geom=foot_orphans, context="VERTS"
                            )
                        bmesh.ops.recalc_face_normals(
                            foot_bm, faces=foot_bm.faces[:]
                        )

                        foot_mesh = bpy.data.meshes.new("Connector")
                        foot_bm.to_mesh(foot_mesh)
                        foot_bm.free()

                        body_bm = split_by_feature_faces(
                            obj,
                            selected_faces_set,
                            boundary_loops,
                            keep_feature=False,
                        )
                        body_bm.to_mesh(obj.data)
                        body_bm.free()

                        # Body-side cleanup: boolean-subtract a shallow
                        # prism built by offsetting the boundary loop
                        # outward and extruding it into the body. This
                        # carves clearance around the foot's perimeter
                        # and pulls back any body geometry that leaned
                        # past the cut plane, so the body's interface
                        # is truly planar. local_plane_no was flipped
                        # above to point toward the foot; negating it
                        # gives the direction into the body.
                        off_clr = props.connector_straight_offset_clearance
                        dep_clr = props.connector_straight_depth_clearance
                        if off_clr > 1e-6 or dep_clr > 1e-6:
                            body_dir = -local_plane_no
                            cleanup_bm = build_straight_cut_body_cleanup_bm(
                                boundary_loops_local,
                                body_dir,
                                off_clr,
                                dep_clr,
                            )
                            cleanup_mesh = bpy.data.meshes.new(
                                "_StraightCleanup"
                            )
                            cleanup_bm.to_mesh(cleanup_mesh)
                            cleanup_bm.free()
                            cleanup_obj = bpy.data.objects.new(
                                "_StraightCleanup", cleanup_mesh
                            )
                            cleanup_obj.matrix_world = (
                                obj.matrix_world.copy()
                            )
                            context.collection.objects.link(cleanup_obj)

                            apply_boolean_difference(
                                context,
                                obj,
                                cleanup_obj,
                                props.connector_solver,
                                self.report,
                            )

                        conn_obj = bpy.data.objects.new("Connector", foot_mesh)
                        conn_obj.matrix_world = obj.matrix_world.copy()
                        context.collection.objects.link(conn_obj)

                        bpy.ops.object.select_all(action="DESELECT")
                        conn_obj.select_set(True)
                        obj.select_set(True)
                        context.view_layer.objects.active = conn_obj

                        did_connector = True
                else:
                    # Smooth the feature boundary loop in-plane so the connector
                    # and cutter share a clean, non-jagged perimeter.  2D outward
                    # normals (re)computed from the smoothed coords feed the
                    # planar-offset clearance step inside build_solid_bmesh.
                    if props.connector_smooth_enabled:
                        smooth_boundary_loops(
                            boundary_loops,
                            vert_coords,
                            dir_vec,
                            iterations=props.connector_smooth_iterations,
                        )
                    boundary_normals = boundary_2d_outward_normals(
                        boundary_loops, vert_coords, dir_vec
                    )

                    conn_bm = build_solid_bmesh(
                        face_vert_lists,
                        vert_coords,
                        selected_verts_set,
                        edge_face_count,
                        depth,
                        direction=direction,
                        draft_angle_rad=draft_rad,
                        boundary_2d_normals=boundary_normals,
                    )
                    conn_mesh = bpy.data.meshes.new("Connector")
                    conn_bm.to_mesh(conn_mesh)
                    conn_bm.free()

                    conn_obj = bpy.data.objects.new("Connector", conn_mesh)
                    conn_obj.matrix_world = obj.matrix_world.copy()
                    context.collection.objects.link(conn_obj)

                    cutter_bm = build_solid_bmesh(
                        face_vert_lists,
                        vert_coords,
                        selected_verts_set,
                        edge_face_count,
                        depth,
                        clearance=clearance,
                        direction=direction,
                        draft_angle_rad=draft_rad,
                        forward_clearance=(
                            props.connector_neg_dir_clearance_value
                            if props.connector_neg_dir_clearance
                            else 0.0
                        ),
                        boundary_2d_normals=boundary_normals,
                    )
                    cutter_mesh = bpy.data.meshes.new("_Cutter")
                    cutter_bm.to_mesh(cutter_mesh)
                    cutter_bm.free()

                    cutter_obj = bpy.data.objects.new("_Cutter", cutter_mesh)
                    cutter_obj.matrix_world = obj.matrix_world.copy()
                    context.collection.objects.link(cutter_obj)

                    bpy.ops.object.select_all(action="DESELECT")
                    obj.select_set(True)
                    context.view_layer.objects.active = obj

                    bool_mod = obj.modifiers.new(name="ConnectorSlot", type="BOOLEAN")
                    bool_mod.operation = "DIFFERENCE"
                    bool_mod.object = cutter_obj
                    bool_mod.solver = props.connector_solver
                    try:
                        bool_mod.use_hole_tolerant = True
                    except AttributeError:
                        pass

                    try:
                        bpy.ops.object.modifier_apply(modifier=bool_mod.name)
                    except RuntimeError as e:
                        self.report(
                            {"WARNING"}, f"Boolean had issues: {e}. Check geometry."
                        )

                    cutter_data = cutter_obj.data
                    bpy.data.objects.remove(cutter_obj, do_unlink=True)
                    bpy.data.meshes.remove(cutter_data)

                    bpy.ops.object.select_all(action="DESELECT")
                    conn_obj.select_set(True)
                    obj.select_set(True)
                    context.view_layer.objects.active = conn_obj

                    did_connector = True

                # Save current feature seeds as previous, then reset
                current_seeds = list(obj.get("cpipe_feature_seeds", []))
                if current_seeds:
                    obj["cpipe_prev_feature_seeds"] = current_seeds
                    props.prev_feature_seeds_set = True
                    props.prev_feature_seed_count = len(current_seeds)

                if "cpipe_feature_seeds" in obj:
                    del obj["cpipe_feature_seeds"]
                props.feature_seeds_set = False
                props.feature_seed_count = 0
                props.gradient_threshold = 0.0

        # ================================================================
        # STEP 5 — BOOLEAN (clearance subtract)
        # ================================================================

        if (
            props.boolean_enabled
            and props.boolean_tool is not None
            and props.boolean_target is not None
            and props.boolean_tool != props.boolean_target
            and props.boolean_tool.type == "MESH"
            and props.boolean_target.type == "MESH"
        ):
            if obj.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")

            ok = perform_boolean_subtract(
                context,
                props.boolean_tool,
                props.boolean_target,
                props.boolean_clearance,
                props.boolean_solver,
                self.report,
            )
            if ok:
                did_boolean = True
            else:
                self.report(
                    {"WARNING"},
                    "Boolean step failed. Try a different solver.",
                )

            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            context.view_layer.objects.active = obj

        # ================================================================
        # Restore mode
        # ================================================================

        current_mode = obj.mode
        if original_mode == "EDIT" and current_mode != "EDIT":
            bpy.ops.object.mode_set(mode="EDIT")
        elif original_mode == "OBJECT" and current_mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")

        context.tool_settings.mesh_select_mode = original_mesh_select_mode

        # ================================================================
        # Report
        # ================================================================

        parts = []
        if did_scale:
            parts.append(f"Scaled to {props.target_height_mm:.0f} mm")
        if did_flatten:
            parts.append(f"Bottom flattened ({flatten_info})")
        if did_bottom_conn:
            offset_parts = []
            if abs(props.bottom_conn_offset_x) > 0.005:
                offset_parts.append(f"X{props.bottom_conn_offset_x:+.1f}")
            if abs(props.bottom_conn_offset_y) > 0.005:
                offset_parts.append(f"Y{props.bottom_conn_offset_y:+.1f}")
            offset_str = f" offset {','.join(offset_parts)}" if offset_parts else ""
            parts.append(
                f"Bottom connector: octagon "
                f"{_BOTTOM_CONN_WIDTH} × {_BOTTOM_CONN_DEPTH} mm"
                f"{offset_str}"
            )
        if did_connector:
            if props.connector_straight_cut_enabled:
                parts.append(
                    f"Straight cut ({props.connector_direction}): "
                    f"best-fit plane through boundary"
                )
            else:
                draft_str = (
                    f", {props.connector_draft_angle:.0f}° draft"
                    if props.connector_draft_enabled
                    else ""
                )
                neg_dir_str = (
                    f" +{props.connector_neg_dir_clearance_value:.2f} mm neg-dir"
                    if props.connector_neg_dir_clearance
                    else ""
                )
                parts.append(
                    f"Feature connector: {props.connector_depth:.1f} mm depth "
                    f"({props.connector_direction}), "
                    f"{props.connector_clearance:.2f} mm clearance"
                    f"{neg_dir_str}"
                    f"{draft_str}"
                )
        if did_boolean:
            parts.append(
                f"Boolean: {props.boolean_tool.name} - {props.boolean_clearance:.3f} mm "
                f"-> {props.boolean_target.name}"
            )

        # ================================================================
        # STEP — EXPORT STLs
        # ================================================================

        if props.export_enabled:
            blend_name = os.path.splitext(os.path.basename(bpy.data.filepath))[0]
            if not blend_name:
                self.report({"ERROR"}, "Save the .blend file first before exporting")
                return {"CANCELLED"}

            dash_idx = blend_name.find("-")
            raw_prefix = blend_name[:dash_idx] if dash_idx > 0 else blend_name
            prefix = raw_prefix.rstrip("0123456789").capitalize()

            export_dir = self.directory

            # Remember current selection state
            prev_active = context.view_layer.objects.active
            prev_selected = [o for o in context.scene.objects if o.select_get()]

            for export_obj in context.scene.objects:
                if export_obj.type != "MESH":
                    continue
                if not props.export_hidden:
                    if export_obj.hide_viewport or export_obj.hide_get():
                        continue

                bpy.ops.object.select_all(action="DESELECT")
                export_obj.select_set(True)
                context.view_layer.objects.active = export_obj

                part_name = export_obj.name.title()
                filename = f"{prefix} - {part_name}.stl"
                filepath = os.path.join(export_dir, filename)

                bpy.ops.wm.stl_export(
                    filepath=filepath,
                    export_selected_objects=True,
                    apply_modifiers=True,
                    ascii_format=False,
                    forward_axis=props.export_forward_axis,
                    up_axis=props.export_up_axis,
                )
                export_count += 1

            # Restore selection state
            bpy.ops.object.select_all(action="DESELECT")
            for o in prev_selected:
                o.select_set(True)
            if prev_active:
                context.view_layer.objects.active = prev_active

            did_export = True

        if did_export:
            parts.append(f"Exported {export_count} STL(s)")

        if parts:
            self.report({"INFO"}, "Pipeline complete! " + " | ".join(parts))
        else:
            self.report({"WARNING"}, "Nothing to do - configure at least one step")
            return {"CANCELLED"}

        return {"FINISHED"}


# ===========================================================================
# Panel
# ===========================================================================


class CPIPE_PT_main(bpy.types.Panel):
    bl_label = "Fuzzin Pipeline"
    bl_idname = "CPIPE_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Fuzzin Pipeline"

    def draw(self, context):
        layout = self.layout
        props = context.scene.cpipe
        obj = context.active_object

        # ---- Scale to Height ----
        box = layout.box()
        row = box.row()
        row.prop(props, "scale_enabled", icon="FIXED_SIZE")
        if props.scale_enabled:
            if props.scale_points_set:
                scale_len = context.scene.unit_settings.scale_length
                dist_mm = props.ref_distance * scale_len * 1000
                box.label(
                    text=f"Current Z height: {dist_mm:.2f} mm", icon="DRIVER_DISTANCE"
                )
            else:
                box.label(text="Select 2 vertices, then set", icon="INFO")

            row = box.row(align=True)
            row.operator("cpipe.set_scale_points", icon="EYEDROPPER")
            row.operator("cpipe.clear_scale_points", text="", icon="X")

            row = box.row(align=True)
            row.enabled = props.prev_scale_points_set
            row.operator("cpipe.restore_scale_points", icon="LOOP_BACK")

            box.prop(props, "target_height_mm")

        # ---- Flatten Bottom ----
        box = layout.box()
        row = box.row()
        row.prop(props, "flatten_bottom_enabled", icon="MOD_LATTICE")
        if props.flatten_bottom_enabled:
            col = box.column(align=True)
            col.prop(props, "flatten_zone_height")
            col.separator()
            col.prop(props, "flatten_solver")

        # ---- Bottom Female Connector ----
        box = layout.box()
        row = box.row()
        row.prop(props, "bottom_connector_enabled", icon="SELECT_SUBTRACT")
        if props.bottom_connector_enabled:
            box.label(
                text=f"Octagon {_BOTTOM_CONN_WIDTH} × {_BOTTOM_CONN_DEPTH} mm",
                icon="INFO",
            )
            col = box.column(align=True)
            col.label(text="Position Offset:")
            row = col.row(align=True)
            row.prop(props, "bottom_conn_offset_x", text="X")
            row.prop(props, "bottom_conn_offset_y", text="Y")

        # ---- Connectors for Features ----
        box = layout.box()
        row = box.row()
        row.prop(props, "feature_connector_enabled", icon="MOD_SOLIDIFY")
        if props.feature_connector_enabled:
            # -- Auto-detect section --
            box.label(text="Auto Mode:", icon="VIEWZOOM")
            box.label(text="Max Gradient Angle Range:")
            row = box.row(align=True)
            row.prop(props, "gradient_range_min", text="Min")
            row.prop(props, "gradient_range_max", text="Max")

            box.prop(props, "gradient_threshold")
            if props.gradient_threshold < 0.5:
                box.label(text="Will auto-detect when run", icon="INFO")

            col = box.column()
            col.scale_y = 1.2
            can_auto = obj and obj.type == "MESH" and obj.mode == "EDIT"
            col.enabled = bool(can_auto)
            col.operator(
                "cpipe.set_feature_seeds",
                text="Auto-detect Vertices",
                icon="EYEDROPPER",
            )

            box.separator()

            # -- Set feature vertices --
            col = box.column()
            col.scale_y = 1.2
            can_set = obj and obj.type == "MESH" and obj.mode == "EDIT"
            col.enabled = bool(can_set)
            col.operator(
                "cpipe.feature_select",
                text="Set Feature Vertices",
                icon="CHECKMARK",
            )

            if props.feature_seeds_set:
                row = box.row(align=True)
                row.label(
                    text=f"{props.feature_seed_count} vertices stored",
                    icon="CHECKMARK",
                )
                row.operator("cpipe.clear_feature_seeds", text="", icon="X")

                row = box.row(align=True)
                row.enabled = props.prev_feature_seeds_set
                row.operator("cpipe.restore_feature_seeds", icon="LOOP_BACK")
            else:
                box.label(text="No feature vertices set", icon="INFO")

            box.separator()

            col = box.column(align=True)
            col.prop(props, "connector_direction")
            col.prop(props, "connector_straight_cut_enabled")

            straight_col = col.column(align=True)
            straight_col.enabled = props.connector_straight_cut_enabled
            straight_col.prop(props, "connector_straight_offset_clearance")
            straight_col.prop(props, "connector_straight_depth_clearance")

            extrude_col = col.column(align=True)
            extrude_col.enabled = not props.connector_straight_cut_enabled
            extrude_col.prop(props, "connector_depth")
            extrude_col.prop(props, "connector_clearance")
            extrude_col.prop(props, "connector_neg_dir_clearance")
            sub = extrude_col.row()
            sub.enabled = props.connector_neg_dir_clearance
            sub.prop(props, "connector_neg_dir_clearance_value")
            extrude_col.prop(props, "connector_draft_enabled")
            sub = extrude_col.row()
            sub.enabled = props.connector_draft_enabled
            sub.prop(props, "connector_draft_angle")
            extrude_col.prop(props, "connector_smooth_enabled")
            sub = extrude_col.row()
            sub.enabled = props.connector_smooth_enabled
            sub.prop(props, "connector_smooth_iterations")

            col.prop(props, "connector_solver")

        # ---- Boolean ----
        box = layout.box()
        row = box.row()
        row.prop(props, "boolean_enabled", icon="MOD_BOOLEAN")
        if props.boolean_enabled:
            col = box.column(align=True)
            col.prop(props, "boolean_tool")
            col.prop(props, "boolean_target")
            if (
                props.boolean_tool is not None
                and props.boolean_target is not None
                and props.boolean_tool == props.boolean_target
            ):
                box.label(text="Tool and target must differ", icon="ERROR")
            box.prop(props, "boolean_clearance")
            box.prop(props, "boolean_solver")

        # ---- Mark Left / Right ----
        box = layout.box()
        row = box.row()
        row.prop(props, "mark_left_right_enabled", icon="FONT_DATA")
        if props.mark_left_right_enabled:
            box.label(
                text=(
                    f"{_TMARK_WIDTH:.0f} × {_TMARK_HEIGHT:.0f} × {_TMARK_DEPTH:.1f} mm shape"
                ),
                icon="INFO",
            )
            box.label(
                text=(f"{_TMARK_LINE:.1f} mm stroke"),
                icon="INFO",
            )
            if obj and obj.type == "MESH":
                islands = _find_mesh_islands(obj)
                if len(islands) >= 2:
                    box.label(
                        text=f"{len(islands)} bodies detected",
                        icon="CHECKMARK",
                    )
                else:
                    box.label(
                        text="Single body",
                        icon="MESH_DATA",
                    )
            box.prop(props, "mark_direction")
            box.prop(props, "mark_solver")
            col = box.column(align=True)
            col.label(text="Position Offset:")
            row = col.row(align=True)
            row.prop(props, "mark_offset_x", text="X")
            row.prop(props, "mark_offset_y", text="Y")
            box.separator()
            row = box.row(align=True)
            row.scale_y = 1.4
            can_mark = obj and obj.type == "MESH"
            row.enabled = bool(can_mark)
            op_l = row.operator("cpipe.mark_side", text="Mark Left", icon="TRIA_LEFT")
            op_l.side = "LEFT"
            op_r = row.operator("cpipe.mark_side", text="Mark Right", icon="TRIA_RIGHT")
            op_r.side = "RIGHT"

        # ---- Export STLs ----
        box = layout.box()
        row = box.row()
        row.prop(props, "export_enabled", icon="EXPORT")
        if props.export_enabled:
            col = box.column(align=True)
            col.prop(props, "export_hidden")
            col.prop(props, "export_forward_axis")
            col.prop(props, "export_up_axis")

        # ---- Run Pipeline ----
        layout.separator()
        col = layout.column()
        col.scale_y = 1.8

        can_run = (
            obj
            and obj.type == "MESH"
            and (
                (props.scale_enabled and props.scale_points_set)
                or props.flatten_bottom_enabled
                or props.bottom_connector_enabled
                or (props.feature_connector_enabled and props.feature_seeds_set)
                or (
                    props.boolean_enabled
                    and props.boolean_tool is not None
                    and props.boolean_target is not None
                    and props.boolean_tool != props.boolean_target
                )
                or props.export_enabled
            )
        )

        if not can_run:
            layout.label(text="Configure at least one step above", icon="ERROR")

        col.enabled = bool(can_run)
        col.operator("cpipe.run_pipeline", text="Run Pipeline", icon="PLAY")


# ===========================================================================
# Registration
# ===========================================================================

classes = (
    CPIPE_Props,
    CPIPE_OT_set_scale_points,
    CPIPE_OT_clear_scale_points,
    CPIPE_OT_restore_scale_points,
    CPIPE_OT_set_feature_seeds,
    CPIPE_OT_clear_feature_seeds,
    CPIPE_OT_restore_feature_seeds,
    CPIPE_OT_feature_select,
    CPIPE_OT_mark_side,
    CPIPE_OT_boolean,
    CPIPE_OT_run_pipeline,
    CPIPE_PT_main,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.cpipe = bpy.props.PointerProperty(type=CPIPE_Props)


def unregister():
    del bpy.types.Scene.cpipe
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
