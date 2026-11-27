# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Native Newton workloads, threaded assets, and small numerical test fixtures."""

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from pxr import Gf, Usd, UsdGeom


@wp.kernel
def drive_particles(
    q: wp.array(dtype=wp.vec3),
    qd: wp.array(dtype=wp.vec3),
    rest: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    angle: float,
    speed: float,
):
    i = wp.tid()
    if inv_mass[i] == 0.0:
        p = rest[i]
        angle = angle * (2.0 * p[0] - 0.5)
        speed = speed * (2.0 * p[0] - 0.5)
        y = p[1] - 0.25
        q[i] = wp.vec3(p[0], 0.25 + y * wp.cos(angle), p[2] + y * wp.sin(angle))
        qd[i] = wp.vec3(0.0, -y * wp.sin(angle) * speed, y * wp.cos(angle) * speed)


@wp.kernel
def drive_bodies(
    q: wp.array(dtype=wp.transform),
    qd: wp.array(dtype=wp.spatial_vector),
    rest: wp.array(dtype=wp.transform),
    inv_mass: wp.array(dtype=float),
    angle: float,
    speed: float,
):
    i = wp.tid()
    if inv_mass[i] == 0.0:
        old = rest[i]
        whip = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * angle)
        twist = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle)
        q[i] = wp.transform(
            wp.transform_get_translation(old), wp.mul(wp.mul(whip, twist), wp.transform_get_rotation(old))
        )
        omega = wp.quat_rotate(whip, wp.vec3(speed, 0.0, 0.0)) + wp.vec3(0.0, 0.0, 0.5 * speed)
        qd[i] = wp.spatial_vector(wp.vec3(0.0), omega)


@wp.kernel
def load_bodies(force: wp.array(dtype=wp.spatial_vector), inv_mass: wp.array(dtype=float), load: wp.vec3):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        force[i] = wp.spatial_vector(load, wp.vec3(0.0))


@wp.kernel
def turn_nut(
    force: wp.array(dtype=wp.spatial_vector),
    velocity: wp.array(dtype=wp.spatial_vector),
    inv_mass: wp.array(dtype=float),
    speed: float,
    gain: float,
    limit: float,
    integral: wp.array(dtype=float),
    integral_gain: float,
    dt: float,
):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        error = speed - velocity[i][5]
        candidate = wp.clamp(integral[i] + integral_gain * dt * error, -limit, limit)
        command = gain * error + candidate
        # Anti-windup retains a bounded motor torque during contact-induced stalls.
        if wp.abs(command) <= limit or command * error < 0.0:
            integral[i] = candidate
        torque = wp.clamp(gain * error + integral[i], -limit, limit)
        force[i] = wp.spatial_vector(wp.spatial_top(force[i]), wp.vec3(0.0, 0.0, torque))


@wp.kernel
def drive_joints(target: wp.array(dtype=float), velocity: wp.array(dtype=float), time: float):
    i = wp.tid()
    phase = 8.0 * time + float(i) * 0.5
    target[i] = 0.6 * wp.sin(phase)
    velocity[i] = 4.8 * wp.cos(phase)


@wp.kernel
def drive_robot(
    target: wp.array(dtype=float),
    velocity: wp.array(dtype=float),
    rest: wp.array(dtype=float),
    amplitude: wp.array(dtype=float),
    frequency: float,
    time: float,
):
    i = wp.tid()
    ramp = wp.min(time / 0.5, 1.0)
    target[i] = rest[i] + ramp * amplitude[i] * wp.sin(frequency * time)
    velocity[i] = amplitude[i] * (
        ramp * frequency * wp.cos(frequency * time) + wp.where(time < 0.5, 2.0, 0.0) * wp.sin(frequency * time)
    )


@wp.kernel
def drive_pusher(
    pose: wp.array(dtype=wp.transform),
    velocity: wp.array(dtype=wp.spatial_vector),
    rest: wp.array(dtype=wp.transform),
    body: int,
    displacement: float,
    speed: float,
):
    position = wp.transform_get_translation(rest[body]) + wp.vec3(displacement, 0.0, 0.0)
    pose[body] = wp.transform(position, wp.transform_get_rotation(rest[body]))
    velocity[body] = wp.spatial_vector(wp.vec3(speed, 0.0, 0.0), wp.vec3(0.0))


@wp.kernel
def load_tool(force: wp.array(dtype=wp.spatial_vector), body: int, load: wp.vec3):
    force[body] = wp.spatial_vector(load, wp.vec3(0.0))


@dataclass
class Scene:
    model: newton.Model
    case: dict
    dt: float
    options: dict
    rest_particles: wp.array | None
    rest_bodies: wp.array | None
    native: object | None = None
    evidence: dict | None = None
    retained_graphs: list = field(default_factory=list)
    joint_amplitude: wp.array | None = None
    load_body: int = -1
    pusher_body: int = -1
    motor_integral: wp.array | None = None

    def prepare(self, state, time, control, *, sampled=False):
        if self.native is not None or sampled:
            # A bank sample includes its native pose, velocity, external forces,
            # and control. Hold that command during the single-timestep probe.
            return
        state.clear_forces()
        if self.joint_amplitude is not None:
            wp.launch(
                drive_robot,
                control.joint_target_q.size,
                [
                    control.joint_target_q,
                    control.joint_target_qd,
                    self.model.joint_target_q,
                    self.joint_amplitude,
                    self.case["drive_frequency"],
                    time,
                ],
            )
            load = np.asarray(self.case["load_direction"], dtype=float) * self.case["load_n"]
            load *= min(time / 0.5, 1.0)
            if self.case["workload"] == "anymal_disturbance":
                load *= np.sin(2 * np.pi * time)
            wp.launch(load_tool, 1, [state.body_f, self.load_body, wp.vec3(*load)])
            return
        if self.case.get("workload") == "rigid_pile":
            phase = np.clip((time - self.case["pusher_start_s"]) / self.case["pusher_duration_s"], 0, 1)
            stroke = self.case["pusher_stroke_m"]
            displacement = stroke * np.sin(np.pi * phase) ** 2
            speed = stroke * np.pi / self.case["pusher_duration_s"] * np.sin(2 * np.pi * phase)
            wp.launch(
                drive_pusher,
                1,
                [state.body_q, state.body_qd, self.rest_bodies, self.pusher_body, displacement, speed],
            )
            return
        if self.case.get("drive", False):
            wp.launch(
                drive_joints, control.joint_target_q.size, [control.joint_target_q, control.joint_target_qd, time]
            )
        amplitude, frequency = self.case.get("amplitude", 0.0), self.case.get("frequency", 2.0)
        angle = amplitude * np.sin(frequency * time)
        speed = amplitude * frequency * np.cos(frequency * time)
        if amplitude and self.model.particle_count:
            wp.launch(
                drive_particles,
                self.model.particle_count,
                [state.particle_q, state.particle_qd, self.rest_particles, self.model.particle_inv_mass, angle, speed],
            )
        if amplitude and self.case["family"] == "cable":
            wp.launch(
                drive_bodies,
                self.model.body_count,
                [state.body_q, state.body_qd, self.rest_bodies, self.model.body_inv_mass, angle, speed],
            )
        if self.case.get("load_n"):
            # A force at each dynamic body's COM; no hidden applied torque.
            direction = np.asarray(self.case.get("load_direction", [0, 0, -1]), dtype=float)
            magnitude = self.case["load_n"] * min(max(time, 0) / self.case.get("ramp_seconds", 0.05), 1)
            wp.launch(
                load_bodies,
                self.model.body_count,
                [state.body_f, self.model.body_inv_mass, wp.vec3(*(direction * magnitude))],
            )
        if self.case.get("thread_speed_rad_s"):
            ramp = min(max(time - 0.2, 0) / 0.2, 1)
            direction = 1 if int(max(time - 0.2, 0) / self.case.get("reverse_seconds", 1e9)) % 2 == 0 else -1
            wp.launch(
                turn_nut,
                self.model.body_count,
                [
                    state.body_f,
                    state.body_qd,
                    self.model.body_inv_mass,
                    self.case["thread_speed_rad_s"] * ramp * direction,
                    self.case.get("torque_gain", 0.002),
                    self.case.get("torque_limit_nm", 0.05),
                    self.motor_integral,
                    self.case.get("torque_integral_gain", 0.0),
                    self.dt,
                ],
            )


def add_rigid_pile(b, case, rng, cfg):
    """Add a mixed-geometry collision bed and return its visible moving wall."""
    r = 0.035
    bin_cfg = cfg.copy()
    bin_cfg.ke, bin_cfg.kd, bin_cfg.mu = 5e3, 5, 0.6
    b.add_ground_plane(cfg=bin_cfg)
    for axis in (0, 1):
        for sign in (-1, 1):
            pos = [0, 0, 0.20]
            pos[axis] = sign * 0.16
            half = [0.18, 0.18, 0.20]
            half[axis] = 0.02
            transform = wp.transform(wp.vec3(*pos), wp.quat_identity())
            body = -1
            if axis == 0 and sign == -1:
                body = pusher = b.add_body(xform=transform, is_kinematic=True, label="pile_pusher")
                transform = wp.transform_identity()
            b.add_shape_box(
                body,
                xform=transform,
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=bin_cfg,
            )
    # Reuse the native benchmark's actual non-convex bear collision mesh.
    stage = Usd.Stage.Open(str(Path(__file__).parent / "_source/newton/examples/assets/bear.usd"))
    geom = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bear/bear"))
    vertices = np.array(geom.GetPointsAttr().Get(), dtype=np.float32)
    vertices -= (vertices.max(0) + vertices.min(0)) / 2
    vertices *= 2 * r / np.ptp(vertices, axis=0).max()
    mesh = newton.Mesh(vertices, np.asarray(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32))
    for i in range(case["count"]):
        x, y, z = i % 3, (i // 3) % 3, i // 9
        pos = np.array([(x - 1) * 0.085, (y - 1) * 0.085, 0.06 + z * 0.085])
        pos[:2] += rng.uniform(-0.002, 0.002, 2)
        body = b.add_body(
            xform=wp.transform(
                wp.vec3(*pos), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), float(rng.uniform(-np.pi, np.pi)))
            ),
            label=f"pile_{i}",
        )
        shape_cfg = cfg.copy()
        shape_cfg.density, shape_cfg.ke, shape_cfg.kd = float(rng.uniform(500, 2500)), 5e3, 5
        kind = i % 6
        if kind == 0:
            b.add_shape_mesh(body, mesh=mesh, cfg=shape_cfg)
        elif kind == 1:
            b.add_shape_cone(body, radius=r, half_height=r, cfg=shape_cfg)
        elif kind == 2:
            b.add_shape_sphere(body, radius=r, cfg=shape_cfg)
        elif kind == 3:
            b.add_shape_box(body, hx=r * 0.7, hy=r * 0.7, hz=r * 0.7, cfg=shape_cfg)
        elif kind == 4:
            b.add_shape_capsule(body, radius=r * 0.5, half_height=r * 0.5, cfg=shape_cfg)
        else:
            b.add_shape_cylinder(body, radius=r, half_height=r * 0.7, cfg=shape_cfg)
    return pusher


def add_robot(b, case, cfg):
    """Import a dynamic mechanism and return its drive amplitudes and loaded link."""
    b.default_shape_cfg = cfg.copy()
    b.default_shape_cfg.ke = 1e5
    if case["workload"] == "franka_payload":
        path = Path(__file__).parent / "assets/robots/franka_emika_panda"
        b.add_urdf(
            path / "urdf/fr3_franka_hand.urdf",
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            collapse_fixed_joints=True,
        )
        rest = [0, -0.5, 0, -2.2, 0, 1.7, 0.8, 0.035, 0.035]
        b.joint_q[:9] = rest
        b.joint_target_q[:9] = rest
        joint_amplitude = [0.7, 0.25, 0.3, 0.3, 0.25, 0.3, 0.4, 0.015, 0.015]
        b.joint_target_ke[:] = [600] * 7 + [200] * 2
        b.joint_target_kd[:] = [25] * 7 + [5] * 2
        load_body = next(i for i, label in enumerate(b.body_label) if label.endswith("fr3_link7"))
        # A real tool mass moves dynamically with the wrist under external load.
        tool_cfg = cfg.copy()
        tool_cfg.density = 4000
        b.add_shape_box(
            load_body,
            xform=wp.transform(wp.vec3(0, 0, 0.227), wp.quat_identity()),
            hx=0.06,
            hy=0.06,
            hz=0.025,
            cfg=tool_cfg,
        )
    else:
        path = Path(__file__).parent / "assets/robots/anybotics_anymal_d"
        b.add_usd(
            str(path / "usd/anymal_d.usda"),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=False,
        )
        b.joint_q[:7] = [0, 0, 0.60, 0, 0, 0, 1]
        joint_amplitude = [0.0] * b.joint_dof_count
        for j, kind in enumerate(b.joint_type):
            if kind == newton.JointType.REVOLUTE:
                dof, coord = b.joint_qd_start[j], b.joint_q_start[j]
                b.joint_target_q[dof] = b.joint_q[coord]
                joint_amplitude[dof] = 0.18 if "KFE" in b.joint_label[j] else 0.12
                b.joint_target_ke[dof], b.joint_target_kd[dof] = 200, 8
        load_body = next(i for i, label in enumerate(b.body_label) if label.endswith("base"))
        # Unequal foot support heights create coupled ground reactions.
        for x in (-0.3, 0.3):
            b.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(x, 0.2, 0.012), wp.quat_identity()),
                hx=0.16,
                hy=0.12,
                hz=0.012,
                cfg=b.default_shape_cfg,
            )
    joint_amplitude = np.asarray(joint_amplitude, dtype=np.float32) * case.get("motion_scale", 1)
    for j, kind in enumerate(b.joint_type):
        if kind in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
            b.joint_target_mode[b.joint_qd_start[j]] = int(newton.JointTargetMode.POSITION)
    b.add_ground_plane(cfg=b.default_shape_cfg)
    return joint_amplitude, load_body


def add_threaded_pair(b, case, scale):
    """Build authored thread contact geometry and the optional axially free spindle."""
    # Preserve authored threaded collision meshes and their world alignment.
    # A voxel is <= 0.1 mm, substantially smaller than the 1.75 mm pitch.
    for part in ("bolt", "nut"):
        stage = Usd.Stage.Open(str(Path(__file__).parent / "assets" / f"{part}_m12.usda"))
        meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and p.HasAPI("PhysicsCollisionAPI")]
        if len(meshes) != 1:
            raise ValueError(f"Expected one authored collision mesh for {part}, got {len(meshes)}")
        geom = UsdGeom.Mesh(meshes[0])
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(meshes[0])
        vertices = np.array([transform.Transform(Gf.Vec3d(*v)) for v in geom.GetPointsAttr().Get()], dtype=np.float32)
        counts, indices = geom.GetFaceVertexCountsAttr().Get(), geom.GetFaceVertexIndicesAttr().Get()
        triangles, cursor = [], 0
        for count in counts:
            face = indices[cursor : cursor + count]
            triangles.extend((face[0], face[j], face[j + 1]) for j in range(1, count - 1))
            cursor += count
        mesh = newton.Mesh(vertices, np.array(triangles, dtype=np.int32).ravel())
        mesh.build_sdf(
            target_voxel_size=1e-4,
            narrow_band_range=(-0.001, 0.001),
            margin=0.0001,
            cache_dir=str(Path.home() / ".cache/neio-sdf"),
        )
        metal = newton.ModelBuilder.ShapeConfig(ke=1e7 * scale, kd=0, mu=0.15, margin=0, gap=0.0002, density=7850)
        body = -1
        if part == "nut":
            start = wp.transform(wp.vec3(0, 0, case.get("thread_start_height_m", 0)), wp.quat_identity())
            if case.get("guided", False):
                body = b.add_link(xform=start, label="nut")
                joint = b.add_joint_d6(
                    -1,
                    body,
                    linear_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z)],
                    angular_axes=[newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z)],
                    parent_xform=start,
                    label="axially_free_spindle",
                )
                b.add_articulation([joint])
            else:
                body = b.add_body(xform=start, label="nut")
        b.add_shape_mesh(body, mesh=mesh, cfg=metal)


def make_scene(case, seed, device="cuda:0", worlds=1):
    """Build once per sampled episode. World IDs isolate contacts and normalization."""
    if "example" in case:
        if worlds != 1:
            raise ValueError("Native workloads use one complete scene per process; batch independent state samples")
        module = importlib.import_module(f"newton.examples.{case['example']}")
        args = SimpleNamespace(
            device=device, seed=seed, solver="vbd", iterations=None, substeps=None, vbd_iterations=10
        )
        with wp.ScopedDevice(device):
            native = module.Example(ViewerNull(), args)
            # Some native collision buffers are allocated during first capture.
            # Their allocator graph must outlive later captures that reuse them.
            original_graph = getattr(native, "graph", None)
            if original_graph is not None:
                # Execute allocation nodes before a second graph can reference
                # their pointers. This is one real recorded simulation frame.
                native.step()
            rng = np.random.default_rng(seed)
            material = float(rng.uniform(*case.get("material_range", [1, 1])))
            # The arrays remain owned by the native model. Captured kernels read
            # these same arrays; no copied builder or replacement controller.
            for name in ("tri_materials", "tet_materials"):
                values = getattr(native.model, name)
                if values is not None and values.size:
                    coefficients = values.numpy()
                    coefficients[:, :2] *= material
                    values.assign(coefficients)
            for name in ("shape_material_ke", "joint_target_ke"):
                values = getattr(native.model, name)
                if values is not None and values.size:
                    values.assign(values.numpy() * material)
            if material != 1:
                native.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
            speed = float(rng.uniform(*case.get("speed_range", [1, 1])))
            if case["example"] == "cloth.example_cloth_twist":
                native.rot_angular_velocity *= speed
            if (material != 1 or speed != 1) and hasattr(native, "capture"):
                native.capture()
        options = {
            key: getattr(native.solver, key)
            for key in (
                "particle_enable_self_contact",
                "friction_epsilon",
                "rigid_soft_enable_dat",
                "rigid_joint_linear_ke",
                "rigid_joint_angular_ke",
            )
        }
        if native.model.particle_count:
            options.update(
                particle_self_contact_margin=native.solver.particle_self_contact_margin,
                particle_self_contact_gap=native.solver.particle_self_contact_gap,
            )
        # Headroom for tightly folded cloth; overflow invalidates a bank.
        options.update(particle_vertex_contact_buffer_size=256, particle_edge_contact_buffer_size=512)
        return Scene(
            native.model,
            {
                **case,
                "seed": seed,
                "material_scale": material,
                "speed_scale": speed,
                "native_warmup_frames": int(original_graph is not None),
            },
            native.sim_dt,
            options,
            None,
            None,
            native,
            retained_graphs=[original_graph] if original_graph is not None else [],
        )
    rng = np.random.default_rng(seed)
    family = case["family"]
    b = newton.ModelBuilder(gravity=wp.vec3(0, 0, case.get("gravity", -9.81)))
    newton.solvers.SolverVBD.register_custom_attributes(b)
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    scale = float(case.get("stiffness_scale", 10 ** rng.uniform(-0.5, 0.5)))
    cfg = newton.ModelBuilder.ShapeConfig(
        ke=3e4 * scale, kd=0, mu=float(case.get("friction", rng.uniform(0.2, 1))), margin=0.001, gap=0.003
    )
    joint_amplitude, load_body, pusher_body = None, -1, -1
    for world in range(worlds):
        b.begin_world(label=f"world_{world}")
        material = scale * float(10 ** rng.uniform(-0.2, 0.2))
        if family in ("cloth", "coupled"):
            n = int(case.get("resolution", 8))
            b.add_cloth_grid(
                pos=wp.vec3(0, 0, 0.6),
                rot=wp.quat_identity(),
                vel=wp.vec3(0, 0, 0 if case.get("rest") else float(rng.uniform(-0.2, 0))),
                dim_x=n,
                dim_y=n,
                cell_x=0.5 / n,
                cell_y=0.5 / n,
                mass=float(rng.uniform(0.1, 0.4)) * (0.5 / n) ** 2,
                fix_left=True,
                fix_right=bool(case.get("twist", False)),
                particle_radius=0.004,
                tri_ke=10 * material,
                tri_ka=20 * material,
                tri_kd=0,
                edge_ke=0.001 * material,
                edge_kd=0,
            )
            b.add_ground_plane(cfg=cfg)
            if case.get("obstacle", False) or family == "coupled":
                body = (
                    b.add_body(xform=wp.transform(wp.vec3(0.3, 0.25, 0.36), wp.quat_identity()))
                    if family == "coupled"
                    else -1
                )
                xform = (
                    wp.transform_identity() if body >= 0 else wp.transform(wp.vec3(0.3, 0.25, 0.36), wp.quat_identity())
                )
                b.add_shape_sphere(body, radius=0.16, xform=xform, cfg=cfg)
        elif family == "soft":
            b.add_soft_grid(
                pos=wp.vec3(0, 0, 0.18),
                rot=wp.quat_identity(),
                vel=wp.vec3(0, 0, -1),
                dim_x=4,
                dim_y=3,
                dim_z=3,
                cell_x=0.05,
                cell_y=0.05,
                cell_z=0.05,
                density=500,
                k_mu=3e4 * material,
                k_lambda=3e4 * material,
                k_damp=0,
                particle_radius=0.005,
            )
            b.add_ground_plane(cfg=cfg)
        elif family == "cable":
            n = int(case.get("resolution", 16))
            points = np.stack((np.linspace(0, 0.8, n + 1), np.zeros(n + 1), np.full(n + 1, 0.25)), -1)
            bodies, _ = b.add_rod(
                rod=newton.Rod(points, radius=0.008),
                cfg=cfg,
                stretch_stiffness=1e5 * material,
                shear_stiffness=1e5 * material,
                bend_stiffness=10 * material,
                twist_stiffness=10 * material,
                body_frame_origin="com",
            )
            first = bodies[0]
            b.body_mass[first] = b.body_inv_mass[first] = 0
            b.body_inertia[first] = b.body_inv_inertia[first] = wp.mat33(0)
            b.add_ground_plane(cfg=cfg)
        elif case.get("workload") == "rigid_pile":
            if worlds != 1:
                raise ValueError("Rigid pile workloads require one complete collision bed per process")
            pusher_body = add_rigid_pile(b, case, rng, cfg)
        elif case.get("workload") in ("franka_payload", "anymal_disturbance"):
            if worlds != 1:
                raise ValueError("Robot workloads require one complete mechanism per process")
            joint_amplitude, load_body = add_robot(b, case, cfg)
        elif family == "rigid":
            b.add_ground_plane(cfg=cfg)
            for i in range(int(case.get("count", 5))):
                body = b.add_body(
                    xform=wp.transform(
                        wp.vec3(0.005 * i, 0, (0.3 if case.get("rest") else 0.05) + i * 0.099), wp.quat_identity()
                    )
                )
                body_cfg = cfg.copy()
                body_cfg.density = float(rng.uniform(500, 1500)) * case.get("mass_ratio", 1) ** (i % 2)
                b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=body_cfg)
        elif family == "articulation":
            joints, parent = [], -1
            for i in range(int(case.get("count", 6))):
                body = b.add_link(xform=wp.transform(wp.vec3(0.05 + i * 0.1, 0, 0.7), wp.quat_identity()))
                b.add_shape_box(body, hx=0.05, hy=0.02, hz=0.02, cfg=cfg)
                joints.append(
                    b.add_joint_revolute(
                        parent=parent,
                        child=body,
                        axis=wp.vec3(0, 1, 0),
                        target_ke=10.0 if case.get("drive") else 0.0,
                        target_kd=0.1 if case.get("drive") else 0.0,
                        parent_xform=wp.transform(
                            wp.vec3(0, 0, 0.7) if parent < 0 else wp.vec3(0.05, 0, 0), wp.quat_identity()
                        ),
                        child_xform=wp.transform(wp.vec3(-0.05, 0, 0), wp.quat_identity()),
                        custom_attributes={"vbd:joint_is_hard": 0},
                    )
                )
                parent = body
            b.add_articulation(joints)
            b.add_ground_plane(cfg=cfg)
        elif family == "m12":
            add_threaded_pair(b, case, scale)
        else:
            raise ValueError(f"Unknown scene family {family}")
        b.end_world()
    b.color()
    model = b.finalize(device=device)
    if joint_amplitude is not None:
        # VBD/NeIO measure joint rotations relative to model.body_q and add the
        # rest angles in model.joint_q. Both must describe the authored pose.
        newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = 1e3 * scale, 0, cfg.mu
    options = dict(particle_enable_self_contact=bool(case.get("self_contact", False)))
    if options["particle_enable_self_contact"]:
        options.update(
            particle_self_contact_margin=0.006,
            particle_self_contact_gap=0.006,
            particle_vertex_contact_buffer_size=128,
            particle_edge_contact_buffer_size=256,
        )
    return Scene(
        model,
        {**case, "stiffness_scale": scale, "friction": cfg.mu, "worlds": worlds},
        float(case.get("dt", 1 / 300)),
        options,
        wp.clone(model.particle_q) if model.particle_count else None,
        wp.clone(model.body_q) if model.body_count else None,
        joint_amplitude=wp.array(joint_amplitude, device=device) if joint_amplitude is not None else None,
        load_body=load_body,
        pusher_body=pusher_body,
        motor_integral=wp.zeros(model.body_count, device=device) if case.get("thread_speed_rad_s") else None,
    )
