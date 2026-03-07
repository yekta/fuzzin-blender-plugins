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
from mathutils import Vector
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
        description="Depth behind the furthest -X point (in scene units)",
        default=3.0,
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

    # --- Mark Left / Right ---
    mark_left_right_enabled: BoolProperty(
        name="Mark Left & Right",
        description=(
            "Engrave an L or r mark on the back (-X) face of the active "
            "object to identify left or right parts"
        ),
        default=False,
    )
    mark_solver: EnumProperty(
        name="Solver",
        description="Boolean solver for the mark cut",
        items=[
            ("EXACT", "Exact", "Slower but most accurate (recommended)"),
            ("MANIFOLD", "Manifold", "Good for complex geometry"),
            ("FLOAT", "Float", "Fast, works for simple shapes"),
        ],
        default="EXACT",
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
# Connectors for Features - Set / Clear Feature Vertices
# ===========================================================================


class CPIPE_OT_set_feature_seeds(bpy.types.Operator):
    """Store the currently selected vertices as feature vertices"""

    bl_idname = "cpipe.set_feature_seeds"
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

        seeds = [v for v in bm.verts if v.select]
        if not seeds:
            self.report(
                {"WARNING"}, "Select at least one seed vertex inside the feature"
            )
            return {"CANCELLED"}

        obj["cpipe_feature_seeds"] = [v.index for v in seeds]

        props = context.scene.cpipe
        props.feature_seeds_set = True
        props.feature_seed_count = len(seeds)
        props.gradient_threshold = 0.0

        self.report({"INFO"}, f"Stored {len(seeds)} feature vertices")
        return {"FINISHED"}


class CPIPE_OT_clear_feature_seeds(bpy.types.Operator):
    """Clear the stored feature vertices"""

    bl_idname = "cpipe.clear_feature_seeds"
    bl_label = "Clear Feature Vertices"

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
# Feature Select (standalone preview with redo)
# ===========================================================================


class CPIPE_OT_feature_select(bpy.types.Operator):
    """Preview feature selection from stored vertices - adjust angle in redo panel"""

    bl_idname = "cpipe.feature_select"
    bl_label = "Preview Features"
    bl_options = {"REGISTER", "UNDO"}

    gradient_angle: FloatProperty(
        name="Max Gradient Angle (deg)",
        description="Maximum normal angle (degrees) between neighbouring vertices",
        default=0.0,
        min=0.0,
        max=90.0,
        step=100,
        precision=1,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not (obj and obj.type == "MESH" and obj.mode == "EDIT"):
            return False
        return "cpipe_feature_seeds" in obj

    def invoke(self, context, event):
        props = context.scene.cpipe
        if props.gradient_threshold < 0.5:
            obj = context.active_object
            seed_indices = list(obj.get("cpipe_feature_seeds", []))
            if seed_indices:
                bm = bmesh.from_edit_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                bm.normal_update()
                optimal, _ = detect_optimal_angle(
                    bm,
                    seed_indices,
                    props.gradient_range_min,
                    props.gradient_range_max,
                )
                props.gradient_threshold = optimal
        self.gradient_angle = props.gradient_threshold
        return self.execute(context)

    def execute(self, context):
        obj = context.active_object
        seed_indices = list(obj.get("cpipe_feature_seeds", []))
        if not seed_indices:
            self.report({"WARNING"}, "No feature vertices stored")
            return {"CANCELLED"}

        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.normal_update()

        grad_limit = math.radians(self.gradient_angle)
        context.scene.cpipe.gradient_threshold = self.gradient_angle

        selected, boundary = bfs_feature_fill(bm, seed_indices, grad_limit)
        selected |= boundary

        bpy.ops.mesh.select_all(action="DESELECT")
        bm.verts.ensure_lookup_table()
        for idx in selected:
            bm.verts[idx].select = True
        bm.select_flush(True)
        bmesh.update_edit_mesh(obj.data)

        self.report(
            {"INFO"}, f"Selected {len(selected)} verts (boundary: {len(boundary)})"
        )
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

        # ---- Find back-face centre of the chosen island ----
        min_x, cy, cz = _island_back_face_centre(obj, target_island)

        # ---- Build the T-mark cutter ----
        mark_bm = create_tmark_cutter_bm(side=self.side)

        mark_mesh = bpy.data.meshes.new("_TMarkCutter")
        mark_bm.to_mesh(mark_mesh)
        mark_bm.free()

        mark_obj = bpy.data.objects.new("_TMarkCutter", mark_mesh)
        context.collection.objects.link(mark_obj)

        # Position: the cutter's X=0 sits at min_x, centred on (cy, cz).
        # The cutter extends in +X so it cuts into the part.
        mark_obj.location = Vector((min_x, cy, cz))
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


def build_solid_bmesh(
    face_vert_lists,
    vert_coords,
    selected_verts_set,
    edge_face_count,
    depth,
    clearance=0.0,
):
    min_x = min(vert_coords[vi].x for vi in selected_verts_set)
    back_x = min_x - depth

    bm = bmesh.new()
    front_map = {}
    back_map = {}

    for vi in selected_verts_set:
        co = vert_coords[vi]
        front_v = bm.verts.new(co)
        front_map[vi] = front_v
        back_co = Vector((back_x, co.y, co.z))
        back_v = bm.verts.new(back_co)
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

    if clearance > 0.0:
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        for f in bm.faces:
            f.normal_update()
        for v in bm.verts:
            if not v.link_faces:
                continue
            avg_normal = Vector((0, 0, 0))
            for f in v.link_faces:
                avg_normal += f.normal
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
# Run Pipeline
# ===========================================================================


class CPIPE_OT_run_pipeline(bpy.types.Operator):
    """Run the pipeline: Scale -> Flatten Bottom -> Bottom Connector -> Feature Connectors"""

    bl_idname = "cpipe.run_pipeline"
    bl_label = "Run Pipeline"
    bl_options = {"REGISTER", "UNDO"}

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
        )

    def execute(self, context):
        obj = context.active_object
        props = context.scene.cpipe
        did_scale = False
        did_flatten = False
        did_bottom_conn = False
        did_connector = False
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

            seed_indices = list(obj.get("cpipe_feature_seeds", []))
            if not seed_indices:
                self.report(
                    {"WARNING"}, "No feature vertices stored - skipping connector"
                )
            else:
                bm = bmesh.from_edit_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                bm.normal_update()

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
                bpy.ops.mesh.select_mode(type="FACE")
                for idx in selected:
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

                conn_bm = build_solid_bmesh(
                    face_vert_lists,
                    vert_coords,
                    selected_verts_set,
                    edge_face_count,
                    depth,
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
            parts.append(
                f"Feature connector: {props.connector_depth:.1f} mm depth, "
                f"{props.connector_clearance:.2f} mm clearance"
            )

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
            if props.feature_seeds_set:
                box.label(
                    text=f"{props.feature_seed_count} vertices stored", icon="CHECKMARK"
                )
            else:
                box.label(text="Select vertices, then set", icon="INFO")

            row = box.row(align=True)
            row.operator("cpipe.set_feature_seeds", icon="EYEDROPPER")
            row.operator("cpipe.clear_feature_seeds", text="", icon="X")

            row = box.row(align=True)
            row.enabled = props.prev_feature_seeds_set
            row.operator("cpipe.restore_feature_seeds", icon="LOOP_BACK")

            box.separator()
            box.label(text="Max Gradient Angle Range:", icon="VIEWZOOM")
            row = box.row(align=True)
            row.prop(props, "gradient_range_min", text="Min")
            row.prop(props, "gradient_range_max", text="Max")

            box.prop(props, "gradient_threshold")
            if props.gradient_threshold < 0.5:
                box.label(text="Will auto-detect when run", icon="INFO")

            box.separator()

            col = box.column()
            col.scale_y = 1.2
            can_select = (
                obj
                and obj.type == "MESH"
                and obj.mode == "EDIT"
                and props.feature_seeds_set
            )
            col.enabled = bool(can_select)
            col.operator(
                "cpipe.feature_select", text="Preview Features", icon="HIDE_OFF"
            )

            box.separator()

            col = box.column(align=True)
            col.prop(props, "connector_depth")
            col.prop(props, "connector_clearance")
            col.prop(props, "connector_solver")

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
            box.prop(props, "mark_solver")
            box.separator()
            row = box.row(align=True)
            row.scale_y = 1.4
            can_mark = obj and obj.type == "MESH"
            row.enabled = bool(can_mark)
            op_l = row.operator("cpipe.mark_side", text="Mark Left", icon="TRIA_LEFT")
            op_l.side = "LEFT"
            op_r = row.operator("cpipe.mark_side", text="Mark Right", icon="TRIA_RIGHT")
            op_r.side = "RIGHT"

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
