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
):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        torque = wp.clamp(gain * (speed - velocity[i][5]), -limit, limit)
        force[i] = wp.spatial_vector(wp.spatial_top(force[i]), wp.vec3(0.0, 0.0, torque))


@wp.kernel
def drive_joints(target: wp.array(dtype=float), velocity: wp.array(dtype=float), time: float):
    i = wp.tid()
    phase = 8.0 * time + float(i) * 0.5
    target[i] = 0.6 * wp.sin(phase)
    velocity[i] = 4.8 * wp.cos(phase)


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

    def prepare(self, state, time, control):
        if self.native is not None:
            # A bank sample includes its native pose, velocity, external forces,
            # and control. Hold that command during the single-timestep probe.
            return
        state.clear_forces()
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
            wp.launch(
                turn_nut,
                self.model.body_count,
                [
                    state.body_f,
                    state.body_qd,
                    self.model.body_inv_mass,
                    self.case["thread_speed_rad_s"] * ramp,
                    0.002,
                    self.case.get("torque_limit_nm", 0.05),
                ],
            )


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
            # Preserve authored threaded collision meshes and their world alignment.
            # A voxel is <= 0.1 mm, substantially smaller than the 1.75 mm pitch.
            for part in ("bolt", "nut"):
                stage = Usd.Stage.Open(str(Path(__file__).parent / "assets" / f"{part}_m12.usda"))
                meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and p.HasAPI("PhysicsCollisionAPI")]
                if len(meshes) != 1:
                    raise ValueError(f"Expected one authored collision mesh for {part}, got {len(meshes)}")
                geom = UsdGeom.Mesh(meshes[0])
                transform = UsdGeom.XformCache().GetLocalToWorldTransform(meshes[0])
                vertices = np.array(
                    [transform.Transform(Gf.Vec3d(*v)) for v in geom.GetPointsAttr().Get()], dtype=np.float32
                )
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
                metal = newton.ModelBuilder.ShapeConfig(
                    ke=1e7 * scale, kd=0, mu=0.15, margin=0, gap=0.0002, density=7850
                )
                body = b.add_body(label="nut") if part == "nut" else -1
                b.add_shape_mesh(body, mesh=mesh, cfg=metal)
        else:
            raise ValueError(f"Unknown scene family {family}")
        b.end_world()
    b.color()
    model = b.finalize(device=device)
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
    )
