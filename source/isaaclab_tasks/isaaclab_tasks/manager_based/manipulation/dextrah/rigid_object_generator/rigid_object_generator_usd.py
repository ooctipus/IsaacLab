from pxr import Usd, UsdGeom, Gf, UsdPhysics
from isaaclab.sim.schemas import schemas, schemas_cfg
import numpy as np
import random

def usd_make_box(stage, path, size, center):
    """
    Create a rectangular cuboid primitive of given size at the specified center.

    - size: (sx, sy, sz)
    - center: (cx, cy, cz)
    """
    # Xform for positioning
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3f(*center))
    sx, sy, sz = size
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    cube = UsdGeom.Cube.Define(stage, path + "/Cube")
    # Default cube size is 2 units; scale by half-extents to match desired dimensions
    cube.AddScaleOp().Set(Gf.Vec3f(hx, hy, hz))
    return cube



def usd_make_cylinder(stage, path, radius, height, center):
    """
    Create a cylinder of given radius & height, centered at 'center'.
    """
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTranslateOp().Set(Gf.Vec3f(*center))
    cyl = UsdGeom.Cylinder.Define(stage, path + "/Cylinder")
    cyl.GetRadiusAttr().Set(radius)
    cyl.GetHeightAttr().Set(height)


# Divider & border builders

def usd_make_horizontal_dividers(stage, parent, width, depth, height, thickness, rows):
    for r in range(1, rows):
        z = height * r / rows
        usd_make_box(
            stage,
            f"{parent}/hDivider_{r}",
            size=(width, depth, thickness),
            center=(width/2, depth/2, z)
        )


def usd_make_vertical_dividers(stage, parent, width, depth, height, thickness, columns):
    for c in range(1, columns):
        x = width * c / columns
        usd_make_box(
            stage,
            f"{parent}/vDivider_{c}",
            size=(thickness, depth, height),
            center=(x, depth/2, height/2)
        )


def usd_make_border(stage, parent, width, depth, height, thickness, open_sides=None):
    if open_sides is None:
        open_sides = []
    # bottom
    if 'bottom' not in open_sides:
        usd_make_box(stage, f"{parent}/border_bottom", (width, depth, thickness),
                     (width/2, depth/2, thickness/2))
    # top
    if 'top' not in open_sides:
        usd_make_box(stage, f"{parent}/border_top", (width, depth, thickness),
                     (width/2, depth/2, height - thickness/2))
    # back
    if 'back' not in open_sides:
        usd_make_box(stage, f"{parent}/border_back", (width, thickness, height),
                     (width/2, thickness/2, height/2))
    # front
    # if 'front' not in open_sides:
    #     usd_make_box(stage, f"{parent}/border_front", (width, thickness, height),
    #                  (width/2, depth - thickness/2, height/2))
    # left
    if 'left' not in open_sides:
        usd_make_box(stage, f"{parent}/border_left", (thickness, depth, height),
                     (thickness/2, depth/2, height/2))
    # right
    if 'right' not in open_sides:
        usd_make_box(stage, f"{parent}/border_right", (thickness, depth, height),
                     (width - thickness/2, depth/2, height/2))


def usd_make_rods(stage, parent, width, depth, height, rod_radius, rows, columns, rod_y_scale=1.0):
    xs = np.linspace(0, width, columns+1)
    ys = np.linspace(0, depth, rows+1)
    center_y = depth/2.0
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            y_s = center_y + (y - center_y) * rod_y_scale
            usd_make_cylinder(
                stage,
                f"{parent}/rod_{i}_{j}",
                radius=rod_radius,
                height=height,
                center=(x, y_s, height/2)
            )

def usd_make_random_vertical_dividers(stage, parentPath,
                                      width, depth, height, thickness,
                                      rows, columns):
    row_h = height / rows
    for c in range(1, columns):
        x = width * c / columns
        free = [(0, rows)]  # intervals in [start, end)
        while free:
            start, end = free.pop(0)
            length = end - start
            if length <= 1:
                continue
            # first-column full-height
            if c == 1 and start == 0 and end == rows:
                usd_make_box(stage, f"{parentPath}/vRnd_full_{c}",
                             (thickness, depth, height),
                             (x, depth/2, height/2))
                break
            # span shorter than full interval, leave one-row buffer
            max_span = length - 1
            span = random.randint(1, max_span)
            offset = random.randint(0, length - span)
            span_h = row_h * span
            z_c = row_h*(start + offset) + span_h/2
            usd_make_box(stage, f"{parentPath}/vRnd_{c}_{start+offset}",
                         (thickness, depth, span_h),
                         (x, depth/2, z_c))
            left = (start, start + offset)
            right = (start + offset + span + 1, end)
            if left[1] - left[0] > 1:
                free.append(left)
            if right[1] - right[0] > 1:
                free.append(right)

def usd_make_random_horizontal_dividers(stage, parent, width, depth, height, thickness,
                                        rows, columns,
                                        ensure_full_length_bottom_divider=False):
    col_w = width / columns
    for r in range(1, rows):
        z = height * r / rows
        # bottom row full width if requested
        if ensure_full_length_bottom_divider and r == rows-1:
            usd_make_box(stage, f"{parent}/hRnd_full_{r}",
                         (width, depth, thickness),
                         (width/2, depth/2, z))
            continue
        free = [(0, columns)]  # intervals in [start, end)
        while free:
            start, end = free.pop(0)
            length = end - start
            if length <= 1:
                continue
            # enforce at least one-column buffer
            max_span = length - 1
            span = random.randint(1, max_span)
            offset = random.randint(0, length - span)
            w_span = col_w * span
            x_c = col_w*(start + offset) + w_span/2
            usd_make_box(stage, f"{parent}/hRnd_{r}_{start+offset}",
                         (w_span, depth, thickness),
                         (x_c, depth/2, z))
            # new intervals, leaving buffer of 1
            left = (start, start+offset)
            right = (start+offset+span+1, end)
            if left[1] - left[0] > 1:
                free.append(left)
            if right[1] - right[0] > 1:
                free.append(right)

def usd_build_grid_shelf(stage, parentPath,
                         width, depth, height, thickness, rows, columns,
                         variant='full_border', open_sides=None,
                         random_horizontal=False,
                         random_vertical=False,
                         no_vertical_devider=False):
    if random_horizontal:
        usd_make_random_horizontal_dividers(
            stage, parentPath, width, depth, height, thickness,
            rows, columns, ensure_full_length_bottom_divider=False
        )
    else:
        usd_make_horizontal_dividers(
            stage, parentPath, width, depth, height, thickness, rows
        )

    if not no_vertical_devider:
        if random_vertical:
            usd_make_random_vertical_dividers(
                stage, parentPath, width, depth, height, thickness, rows, columns
            )
        else:
            usd_make_vertical_dividers(
                stage, parentPath, width, depth, height, thickness, columns
            )

    if variant == 'full_border':
        osl = []
    elif variant == 'none':
        osl = ['left','right','front','back','top','bottom']
    else:
        osl = open_sides or []

    usd_make_border(
        stage, parentPath, width, depth, height, thickness, osl
    )

def usd_build_rod_shelf(stage, parentPath,
                        width, depth, height, thickness,
                        rows, columns,
                        rod_radius, rod_y_scale,
                        random_horizontal=False):
    if random_horizontal:
        usd_make_random_horizontal_dividers(
            stage, parentPath, width, depth, height, thickness,
            rows, columns, ensure_full_length_bottom_divider=True
        )
    else:
        usd_make_horizontal_dividers(
            stage, parentPath, width, depth, height, thickness, rows
        )

    usd_make_rods(
        stage, parentPath, width, depth, height,
        rod_radius, 1, columns, rod_y_scale
    )

def usd_build_shelf(stage, parentPath,
                    width, depth, height, thickness, rows, columns):
    presets = [
        { 'type':'grid', 'variant':'one_side_open', 'open_sides':['front'],
          'random_vertical':True,  'random_horizontal':True, 'no_vertical_devider':False },
        { 'type':'grid', 'variant':'one_side_open', 'open_sides':['front'],
          'random_vertical':True,  'random_horizontal':False, 'no_vertical_devider':False },
        { 'type':'grid', 'variant':'one_side_open', 'open_sides':['front'],
          'random_vertical':False, 'random_horizontal':True,  'no_vertical_devider':False },
        { 'type':'grid', 'variant':'one_side_open', 'open_sides':['front'],
          'random_vertical':False, 'random_horizontal':False, 'no_vertical_devider':True  },
        { 'type':'grid', 'variant':'one_side_open', 'open_sides':['front','back'],
          'random_vertical':True,  'random_horizontal':False, 'no_vertical_devider':False },
        { 'type':'grid', 'variant':'one_side_open', 'open_sides':['front','back'],
          'random_vertical':False, 'random_horizontal':True,  'no_vertical_devider':False },
        { 'type':'grid', 'variant':'one_side_open', 'open_sides':['front','back'],
          'random_vertical':False, 'random_horizontal':False, 'no_vertical_devider':True  },
        { 'type':'grid', 'variant':'one_side_open', 'open_sides':['front','left','right'],
          'random_vertical':False, 'random_horizontal':False, 'no_vertical_devider':True  },
        { 'type':'rod',  'random_horizontal':True,  'rod_y_scale_range':(0.5, 1.0) },
        { 'type':'rod',  'random_horizontal':False, 'rod_y_scale_range':(0.5, 1.0) }
    ]
    cfg = random.choice(presets)

    if cfg['type'] == 'grid':
        usd_build_grid_shelf(
            stage, parentPath,
            width, depth, height, thickness, rows, columns,
            variant=cfg['variant'],
            open_sides=cfg.get('open_sides'),
            random_horizontal=cfg['random_horizontal'],
            random_vertical=cfg['random_vertical'],
            no_vertical_devider=cfg['no_vertical_devider']
        )
    else:
        y_scale = random.uniform(*cfg['rod_y_scale_range'])
        usd_build_rod_shelf(
            stage, parentPath,
            width, depth, height, thickness,
            rows=max(rows, 3), columns=columns,
            rod_radius=0.01, rod_y_scale=y_scale,
            random_horizontal=cfg['random_horizontal']
        )

def usd_build_shelf_range(stage, parentPath,
                          length_range, depth_range,
                          height_range, thickness_range,
                          row_range, col_range):
    length = random.uniform(*length_range)
    depth  = random.uniform(*depth_range)
    height = random.uniform(*height_range)
    thickness = random.uniform(*thickness_range)

    # clamp rows/cols by scale thresholds
    def clamp(scale, rng):
        for thr, red in [(0.4,3),(0.6,2),(0.8,1)]:
            if scale < thr:
                return max(rng[0], rng[1] - red)
        return rng[1]

    len_scale = (length - length_range[0])/(length_range[1]-length_range[0])
    hgt_scale = (height - height_range[0])/(height_range[1]-height_range[0])
    max_rows = clamp(hgt_scale, row_range)
    max_cols = clamp(len_scale, col_range)
    rows = random.randint(max(1, row_range[0]), max_rows)
    cols = random.randint(max(2, col_range[0]), max_cols)

    # Build the shelf geometry
    usd_build_shelf(
        stage, parentPath,
        length, depth, height, thickness,
        rows, cols
    )

    # Physics: assign rigid body and mass
    shelfPrim = stage.GetPrimAtPath(parentPath)
    UsdPhysics.RigidBodyAPI.Apply(shelfPrim)
    massAPI = UsdPhysics.MassAPI.Apply(shelfPrim)
    massAPI.GetMassAttr().Set(1.0)

    # Collision: cube, cylinder primitives
    for prim in Usd.PrimRange(shelfPrim):
        t = prim.GetTypeName()
        if t in ("Cube", "Cylinder"):
            schemas.define_collision_properties(
                prim_path=prim.GetPath(),
                cfg=schemas_cfg.CollisionPropertiesCfg(collision_enabled=True),
                stage=stage
            )

def build_shelf_usd(usd_path, length_range, depth_range, height_range, thickness_range, row_range, col_range, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Shelves")
    stage.SetDefaultPrim(root.GetPrim())

    usd_build_shelf_range(
        stage, "/Shelves", length_range, depth_range, height_range, thickness_range, row_range, col_range
    )
    # Save out and close
    stage.GetRootLayer().Save()
    

if __name__ == "__main__":
    for i in range(10):
        build_shelf_usd(
            f"shelf{i}.usd",
            length_range=(0.8, 1.6),
            depth_range=(0.3, 0.4),
            height_range=(0.5, 1.0),
            thickness_range=(0.02, 0.05),
            row_range=(2, 5),
            col_range=(2, 5)
        )