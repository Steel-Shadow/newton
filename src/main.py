import math
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples

DOMINO_COUNT = 96
DOMINO_HALF_THICKNESS = 0.035
DOMINO_HALF_WIDTH = 0.14
DOMINO_HALF_HEIGHT = 0.40
DOMINO_SPACING = 0.36
DOMINO_CIRCLE_ENTRY_ANGLE = -0.5 * math.pi
DOMINO_CIRCLE_ENTRY_GAP = 5.0
FIRST_DOMINO_X = 0.75

BALL_RADIUS = 0.18
BALL_DENSITY = 350.0
RAMP_LENGTH = 3.8
RAMP_WIDTH = 0.9
RAMP_THICKNESS = 0.12

SOFT_BUFFER_DIM_X = 3
SOFT_BUFFER_DIM_Y = 4
SOFT_BUFFER_DIM_Z = 3
SOFT_BUFFER_CELL_X = 0.073
SOFT_BUFFER_CELL_Y = 0.105
SOFT_BUFFER_CELL_Z = 0.113
SOFT_BUFFER_CONTACT_CLEARANCE = 0.03
SOFT_BUFFER_PARTICLE_RADIUS = 0.01
SOFT_BUFFER_PARTICLE_DOMINO_COUNT = 1
SOFT_BUFFER_DENSITY = 100.0
SOFT_BUFFER_K_MU = 1.0e4
SOFT_BUFFER_K_LAMBDA = 5.0e4
SOFT_BUFFER_K_DAMP = 1.0
SOFT_CONTACT_MARGIN = 0.01


@dataclass
class EditableBody:
    label: str
    body: int


@dataclass
class DominoPlacement:
    x: float
    y: float
    yaw: float
    group: str = "pattern"


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        self.domino_body_indices: list[int] = []
        self.chain_body_indices: list[int] = []
        self.editable_bodies: list[EditableBody] = []
        self.selected_edit_index = 0
        self.edit_position = [0.0, 0.0, 0.0]
        self.edit_rotation_deg = [0.0, 0.0, 0.0]
        self.zero_velocity_on_apply = False
        self.ball_body_index: int | None = None
        self.ramp_body_index: int | None = None
        self.soft_particle_indices: np.ndarray | None = None

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_particle_radius = SOFT_BUFFER_PARTICLE_RADIUS
        builder.particle_max_velocity = 80.0

        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.mu = 0.8
        builder.add_ground_plane(cfg=ground_cfg, color=wp.vec3(0.20, 0.22, 0.24), label="ground")

        self._add_ramp(builder, args.ramp_angle)
        self._add_ball(builder, args.ball_speed, args.ramp_angle)
        if not args.disable_soft_buffer:
            self._add_soft_buffer(builder, args)
        self._add_dominoes(builder, args)

        builder.joint_qd = np.array(builder.body_qd).flatten().tolist()

        self.model = builder.finalize()
        self.model.soft_contact_ke = args.soft_contact_ke
        self.model.soft_contact_kd = args.soft_contact_kd
        self.model.soft_contact_kf = args.soft_contact_kf
        self.model.soft_contact_mu = args.soft_contact_mu
        self.model.soft_contact_restitution = args.soft_contact_restitution
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=args.broad_phase,
            soft_contact_margin=args.soft_contact_margin,
        )
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=args.iterations,
            rigid_contact_relaxation=0.85,
            soft_contact_relaxation=0.9,
            angular_damping=0.01,
            enable_restitution=True,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.initial_body_q = self.state_0.body_q.numpy().copy()
        self.initial_body_qd = self.state_0.body_qd.numpy().copy()
        self.initial_particle_q = (
            self.state_0.particle_q.numpy().copy() if self.state_0.particle_q is not None else None
        )
        self.initial_particle_qd = (
            self.state_0.particle_qd.numpy().copy() if self.state_0.particle_qd is not None else None
        )
        self._load_edit_pose_from_state()

        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self._set_camera()

        self.capture()

    @staticmethod
    def _ramp_top_z(local_x: float, angle: float, center_z: float) -> float:
        return center_z - math.sin(angle) * local_x + math.cos(angle) * (RAMP_THICKNESS * 0.5)

    def _add_ramp(self, builder: newton.ModelBuilder, ramp_angle_deg: float) -> None:
        self.ramp_angle = math.radians(ramp_angle_deg)
        self.ramp_center_x = -1.75
        self.ramp_low_top_z = BALL_RADIUS + 0.03
        self.ramp_center_z = (
            self.ramp_low_top_z
            + math.sin(self.ramp_angle) * (RAMP_LENGTH * 0.5)
            - math.cos(self.ramp_angle) * (RAMP_THICKNESS * 0.5)
        )

        ramp_cfg = builder.default_shape_cfg.copy()
        ramp_cfg.mu = 0.55
        ramp_cfg.restitution = 0.05
        ramp_cfg.has_particle_collision = False

        ramp_q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), self.ramp_angle)
        self.ramp_body_index = builder.add_body(
            xform=wp.transform(p=wp.vec3(self.ramp_center_x, 0.0, self.ramp_center_z), q=ramp_q),
            label="inclined_ramp",
            is_kinematic=True,
        )
        builder.add_shape_box(
            body=self.ramp_body_index,
            hx=RAMP_LENGTH * 0.5,
            hy=RAMP_WIDTH * 0.5,
            hz=RAMP_THICKNESS * 0.5,
            cfg=ramp_cfg,
            color=wp.vec3(0.78, 0.62, 0.36),
            label="inclined_ramp",
        )
        self.editable_bodies.append(EditableBody("Ramp", self.ramp_body_index))

    def _add_ball(self, builder: newton.ModelBuilder, ball_speed: float, ramp_angle_deg: float) -> None:
        angle = math.radians(ramp_angle_deg)
        local_x = -RAMP_LENGTH * 0.38
        ball_x = self.ramp_center_x + math.cos(angle) * local_x
        ball_z = self._ramp_top_z(local_x, angle, self.ramp_center_z) + BALL_RADIUS + 0.02

        ball_cfg = builder.default_shape_cfg.copy()
        ball_cfg.density = BALL_DENSITY
        ball_cfg.mu = 0.65
        ball_cfg.restitution = 0.15
        ball_cfg.mu_rolling = 0.00002

        self.ball_body_index = builder.add_body(
            xform=wp.transform(p=wp.vec3(ball_x, 0.0, ball_z), q=wp.quat_identity()),
            label="rolling_ball",
        )
        builder.add_shape_ellipsoid(
            self.ball_body_index,
            rx=BALL_RADIUS,
            ry=BALL_RADIUS,
            rz=BALL_RADIUS,
            cfg=ball_cfg,
            color=wp.vec3(0.12, 0.42, 0.95),
            label="rolling_ball_shape",
        )
        self.editable_bodies.append(EditableBody("Ball", self.ball_body_index))

        builder.body_qd[self.ball_body_index] = wp.spatial_vector(
            ball_speed * math.cos(angle),
            0.0,
            -ball_speed * math.sin(angle),
            0.0,
            ball_speed / BALL_RADIUS,
            0.0,
        )

    def _add_soft_buffer(self, builder: newton.ModelBuilder, args) -> None:
        length = SOFT_BUFFER_DIM_X * SOFT_BUFFER_CELL_X
        width = SOFT_BUFFER_DIM_Y * SOFT_BUFFER_CELL_Y
        gap_to_domino = SOFT_BUFFER_PARTICLE_RADIUS + args.soft_contact_margin + SOFT_BUFFER_CONTACT_CLEARANCE
        right_x = FIRST_DOMINO_X - DOMINO_HALF_THICKNESS - gap_to_domino
        start_x = right_x - length

        start = len(builder.particle_q)
        builder.add_soft_grid(
            pos=wp.vec3(start_x, -0.5 * width, SOFT_BUFFER_PARTICLE_RADIUS),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=SOFT_BUFFER_DIM_X,
            dim_y=SOFT_BUFFER_DIM_Y,
            dim_z=SOFT_BUFFER_DIM_Z,
            cell_x=SOFT_BUFFER_CELL_X,
            cell_y=SOFT_BUFFER_CELL_Y,
            cell_z=SOFT_BUFFER_CELL_Z,
            density=args.soft_density,
            k_mu=args.soft_k_mu,
            k_lambda=args.soft_k_lambda,
            k_damp=args.soft_damping,
            tri_ke=args.soft_surface_ke,
            tri_ka=args.soft_surface_ka,
            tri_kd=args.soft_surface_kd,
            particle_radius=SOFT_BUFFER_PARTICLE_RADIUS,
            label="soft_buffer",
        )
        self.soft_particle_indices = np.arange(start, len(builder.particle_q), dtype=np.int32)

    @staticmethod
    def _line_placements(
        count: int,
        *,
        start_x: float,
        start_y: float,
        spacing: float,
        yaw: float = 0.0,
        group: str = "entry",
    ) -> list[DominoPlacement]:
        direction = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
        return [
            DominoPlacement(
                x=start_x + direction[0] * spacing * i,
                y=start_y + direction[1] * spacing * i,
                yaw=yaw,
                group=group,
            )
            for i in range(count)
        ]

    @staticmethod
    def _circle_placements(
        count: int,
        *,
        center_x: float,
        center_y: float,
        radius: float,
        start_angle: float = math.pi,
        sweep_angle: float = 2.0 * math.pi,
        group: str = "circle",
    ) -> list[DominoPlacement]:
        if count <= 0:
            return []

        placements = []
        for i in range(count):
            if abs(sweep_angle - 2.0 * math.pi) < 1.0e-6:
                angle = start_angle + sweep_angle * i / count
            else:
                angle = start_angle + sweep_angle * i / max(count - 1, 1)
            placements.append(
                DominoPlacement(
                    x=center_x + radius * math.cos(angle),
                    y=center_y + radius * math.sin(angle),
                    yaw=angle + math.pi * 0.5,
                    group=group,
                )
            )
        return placements

    @staticmethod
    def _wave_placements(
        count: int,
        *,
        start_x: float,
        start_y: float,
        spacing: float,
        amplitude: float,
        wavelength: float,
        group: str = "wave",
    ) -> list[DominoPlacement]:
        placements = []
        for i in range(count):
            x = start_x + spacing * i
            phase = (x - start_x) / wavelength * 2.0 * math.pi
            y = start_y + amplitude * math.sin(phase)
            slope = amplitude * (2.0 * math.pi / wavelength) * math.cos(phase)
            placements.append(DominoPlacement(x=x, y=y, yaw=math.atan2(slope, 1.0), group=group))
        return placements

    @staticmethod
    def _spiral_placements(
        count: int,
        *,
        center_x: float,
        center_y: float,
        spacing: float,
        start_radius: float,
        radial_step: float,
        group: str = "spiral",
    ) -> list[DominoPlacement]:
        placements = []
        radius = start_radius
        angle = math.pi
        for _ in range(count):
            placements.append(
                DominoPlacement(
                    x=center_x + radius * math.cos(angle),
                    y=center_y + radius * math.sin(angle),
                    yaw=angle + math.pi * 0.5,
                    group=group,
                )
            )
            angle += spacing / max(radius, spacing)
            radius += radial_step
        return placements

    def _build_domino_placements(self, args) -> list[DominoPlacement]:
        count = max(1, args.domino_count)
        spacing = args.domino_spacing
        scale = args.pattern_scale
        first_x = FIRST_DOMINO_X

        if args.domino_pattern == "line":
            return self._line_placements(count, start_x=first_x, start_y=0.0, spacing=spacing)

        if args.domino_pattern == "wave":
            return self._wave_placements(
                count,
                start_x=first_x,
                start_y=0.0,
                spacing=spacing,
                amplitude=0.85 * scale,
                wavelength=3.2 * scale,
                group="entry",
            )

        entry_count = min(max(8, count // 6), count, 16)
        entry = self._line_placements(count=entry_count, start_x=first_x, start_y=0.0, spacing=spacing, group="entry")
        remaining = count - entry_count
        entry_end_x = first_x + (entry_count - 1) * spacing
        circle_gap = DOMINO_CIRCLE_ENTRY_GAP * spacing

        if args.domino_pattern == "circle":
            radius = max(1.0 * scale, ((remaining - 1) * spacing + circle_gap) / (2.0 * math.pi))
            sweep_angle = 2.0 * math.pi - circle_gap / radius
            center_x = entry_end_x + spacing
            return entry + self._circle_placements(
                remaining,
                center_x=center_x,
                center_y=radius,
                radius=radius,
                start_angle=DOMINO_CIRCLE_ENTRY_ANGLE,
                sweep_angle=sweep_angle,
            )

        if args.domino_pattern == "spiral":
            center_x = first_x + (entry_count - 1) * spacing + 1.35 * scale
            return entry + self._spiral_placements(
                remaining,
                center_x=center_x,
                center_y=0.0,
                spacing=spacing,
                start_radius=0.55 * scale,
                radial_step=0.018 * scale,
            )

        circle_count = remaining
        radius = max(1.15 * scale, ((circle_count - 1) * spacing + circle_gap) / (2.0 * math.pi))
        sweep_angle = 2.0 * math.pi - circle_gap / radius
        circle_center_x = entry_end_x + spacing
        circle_center_y = radius
        circle = self._circle_placements(
            circle_count,
            center_x=circle_center_x,
            center_y=circle_center_y,
            radius=radius,
            start_angle=DOMINO_CIRCLE_ENTRY_ANGLE,
            sweep_angle=sweep_angle,
        )
        return entry + circle

    @staticmethod
    def _domino_color(index: int, total: int, group: str) -> wp.vec3:
        if group == "entry":
            return wp.vec3(0.92, 0.18, 0.12) if index == 0 else wp.vec3(0.98, 0.70, 0.18)
        if group == "circle":
            phase = 2.0 * math.pi * index / max(total, 1)
            return wp.vec3(
                0.45 + 0.35 * math.sin(phase),
                0.50 + 0.30 * math.sin(phase + 2.1),
                0.70 + 0.25 * math.sin(phase + 4.2),
            )
        if group == "spiral":
            return wp.vec3(0.25, 0.75, 0.88)
        return wp.vec3(0.72, 0.48, 0.95)

    def _add_dominoes(self, builder: newton.ModelBuilder, args) -> None:
        domino_cfg = builder.default_shape_cfg.copy()
        domino_cfg.density = 250.0
        domino_cfg.mu = 0.9
        domino_cfg.restitution = 0.05

        placements = self._build_domino_placements(args)
        self.domino_positions = np.array([[p.x, p.y] for p in placements], dtype=np.float32)

        for i, placement in enumerate(placements):
            yaw_q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), placement.yaw)
            body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(placement.x, placement.y, DOMINO_HALF_HEIGHT),
                    q=yaw_q,
                ),
                label=f"domino_{i:02d}",
            )
            cfg = domino_cfg.copy()
            cfg.has_particle_collision = i < SOFT_BUFFER_PARTICLE_DOMINO_COUNT and not args.disable_soft_buffer
            builder.add_shape_box(
                body,
                hx=DOMINO_HALF_THICKNESS,
                hy=DOMINO_HALF_WIDTH,
                hz=DOMINO_HALF_HEIGHT,
                cfg=cfg,
                color=self._domino_color(i, len(placements), placement.group),
                label=f"domino_{i:02d}_shape",
            )
            self.domino_body_indices.append(body)
            if placement.group != "wave":
                self.chain_body_indices.append(body)
            self.editable_bodies.append(EditableBody(f"Domino {i:02d}", body))

    def _set_camera(self) -> None:
        if len(self.domino_positions) == 0:
            self.viewer.set_camera(pos=wp.vec3(2.9, -4.6, 2.3), pitch=-22.0, yaw=148.0)
            return

        mins = self.domino_positions.min(axis=0)
        maxs = self.domino_positions.max(axis=0)
        center = 0.5 * (mins + maxs)
        span = float(max(maxs[0] - mins[0], maxs[1] - mins[1], 4.0))
        self.viewer.set_camera(
            pos=wp.vec3(float(center[0] + 0.35 * span), float(center[1] - 1.25 * span), max(2.6, 0.62 * span)),
            pitch=-28.0,
            yaw=138.0,
        )

    @staticmethod
    def _quat_to_rpy_deg(q_values) -> list[float]:
        q = wp.quat(*q_values)
        matrix = np.array(wp.quat_to_matrix(q), dtype=np.float32).reshape(3, 3)
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        pitch = math.atan2(
            -float(matrix[2, 0]),
            math.sqrt(float(matrix[2, 1]) ** 2 + float(matrix[2, 2]) ** 2),
        )
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
        return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

    @staticmethod
    def _rpy_deg_to_quat(rotation_deg: list[float]) -> wp.quat:
        roll, pitch, yaw = (math.radians(v) for v in rotation_deg)
        return wp.quat_rpy(roll, pitch, yaw)

    def _selected_editable_body(self) -> EditableBody:
        self.selected_edit_index = min(max(self.selected_edit_index, 0), len(self.editable_bodies) - 1)
        return self.editable_bodies[self.selected_edit_index]

    def _load_edit_pose_from_state(self) -> None:
        if not self.editable_bodies:
            return
        selected = self._selected_editable_body()
        body_q = self.state_0.body_q.numpy()
        pose = body_q[selected.body]
        self.edit_position = [float(pose[0]), float(pose[1]), float(pose[2])]
        self.edit_rotation_deg = self._quat_to_rpy_deg(pose[3:7])

    def _sync_joint_coordinates(self) -> None:
        newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)
        newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)

    def _refresh_contacts_after_edit(self) -> None:
        self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

    def _apply_edit_pose(self) -> None:
        selected = self._selected_editable_body()
        q = self._rpy_deg_to_quat(self.edit_rotation_deg)

        for state in (self.state_0, self.state_1):
            body_q = state.body_q.numpy()
            body_q[selected.body] = np.array(
                [
                    self.edit_position[0],
                    self.edit_position[1],
                    self.edit_position[2],
                    float(q[0]),
                    float(q[1]),
                    float(q[2]),
                    float(q[3]),
                ],
                dtype=body_q.dtype,
            )
            state.body_q.assign(body_q)

            if self.zero_velocity_on_apply:
                body_qd = state.body_qd.numpy()
                body_qd[selected.body] = np.zeros(6, dtype=body_qd.dtype)
                state.body_qd.assign(body_qd)

        self._sync_joint_coordinates()
        self._refresh_contacts_after_edit()
        self.graph = None

    def _restore_selected_body(self) -> None:
        selected = self._selected_editable_body()
        for state in (self.state_0, self.state_1):
            body_q = state.body_q.numpy()
            body_qd = state.body_qd.numpy()
            body_q[selected.body] = self.initial_body_q[selected.body]
            body_qd[selected.body] = self.initial_body_qd[selected.body]
            state.body_q.assign(body_q)
            state.body_qd.assign(body_qd)

        self._sync_joint_coordinates()
        self._load_edit_pose_from_state()
        self._refresh_contacts_after_edit()
        self.graph = None

    def _restore_scene(self) -> None:
        for state in (self.state_0, self.state_1):
            state.body_q.assign(self.initial_body_q)
            state.body_qd.assign(self.initial_body_qd)
            if self.initial_particle_q is not None and state.particle_q is not None:
                state.particle_q.assign(self.initial_particle_q)
            if self.initial_particle_qd is not None and state.particle_qd is not None:
                state.particle_qd.assign(self.initial_particle_qd)

        self.sim_time = 0.0
        self._sync_joint_coordinates()
        self._load_edit_pose_from_state()
        self._refresh_contacts_after_edit()
        self.graph = None

    def gui(self, ui):
        paused = self.viewer.is_paused() if hasattr(self.viewer, "is_paused") else False

        ui.text("Paused Object Editor")
        if not paused:
            ui.text("Pause the simulation before applying edits.")

        labels = [body.label for body in self.editable_bodies]
        changed, selected_index = ui.combo("Object", self.selected_edit_index, labels)
        if changed:
            self.selected_edit_index = selected_index
            self._load_edit_pose_from_state()

        ui.separator()
        changed, self.edit_position[0] = ui.slider_float("X [m]", self.edit_position[0], -4.0, 6.0, "%.3f")
        changed_y, self.edit_position[1] = ui.slider_float("Y [m]", self.edit_position[1], -2.0, 2.0, "%.3f")
        changed_z, self.edit_position[2] = ui.slider_float("Z [m]", self.edit_position[2], 0.0, 3.0, "%.3f")
        changed = changed or changed_y or changed_z

        changed_roll, self.edit_rotation_deg[0] = ui.slider_float(
            "Roll [deg]", self.edit_rotation_deg[0], -180.0, 180.0, "%.1f"
        )
        changed_pitch, self.edit_rotation_deg[1] = ui.slider_float(
            "Pitch [deg]", self.edit_rotation_deg[1], -180.0, 180.0, "%.1f"
        )
        changed_yaw, self.edit_rotation_deg[2] = ui.slider_float(
            "Yaw [deg]", self.edit_rotation_deg[2], -180.0, 180.0, "%.1f"
        )
        changed = changed or changed_roll or changed_pitch or changed_yaw

        _changed, self.zero_velocity_on_apply = ui.checkbox("Zero velocity on apply", self.zero_velocity_on_apply)

        if paused and changed:
            self._apply_edit_pose()

        if ui.button("Apply Pose"):
            if paused:
                self._apply_edit_pose()
        ui.same_line()
        if ui.button("Reload From Scene"):
            self._load_edit_pose_from_state()

        if ui.button("Restore Object"):
            if paused:
                self._restore_selected_body()
        ui.same_line()
        if ui.button("Restore Scene"):
            if paused:
                self._restore_scene()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        if not np.isfinite(body_q).all() or not np.isfinite(body_qd).all():
            raise ValueError("Simulation produced non-finite rigid body state")

        if self.state_0.particle_q is not None:
            particle_q = self.state_0.particle_q.numpy()
            particle_qd = self.state_0.particle_qd.numpy()
            if not np.isfinite(particle_q).all() or not np.isfinite(particle_qd).all():
                raise ValueError("Simulation produced non-finite soft body state")

            if self.soft_particle_indices is not None:
                soft_q = particle_q[self.soft_particle_indices]
                soft_extent = soft_q.max(axis=0) - soft_q.min(axis=0)
                if np.linalg.norm(soft_extent) > 2.0:
                    raise ValueError("Soft buffer expanded beyond the expected range")

        test_bodies = self.chain_body_indices or self.domino_body_indices
        tilted_count = 0
        for body in test_bodies:
            q = wp.quat(*body_q[body][3:7])
            local_up = np.array(wp.quat_to_matrix(q), dtype=np.float32).reshape(3, 3)[:, 2]
            if abs(float(local_up[2])) < 0.85:
                tilted_count += 1

        required_tilted = max(2, len(test_bodies) // 4)
        if tilted_count < required_tilted:
            raise ValueError(f"Only {tilted_count} dominoes tilted; expected a visible chain reaction")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_broad_phase_arg(parser)
        parser.set_defaults(broad_phase="sap", num_frames=600)
        parser.add_argument("--domino-count", type=int, default=DOMINO_COUNT, help="Total number of dominoes.")
        parser.add_argument(
            "--domino-spacing",
            type=float,
            default=DOMINO_SPACING,
            help="Distance between neighboring domino centers along generated paths [m].",
        )
        parser.add_argument(
            "--domino-pattern",
            type=str,
            default="showcase",
            choices=["showcase", "line", "circle", "spiral", "wave"],
            help="Domino layout pattern.",
        )
        parser.add_argument(
            "--pattern-scale",
            type=float,
            default=1.0,
            help="Scale factor for circular, spiral, and wave pattern dimensions.",
        )
        parser.add_argument(
            "--ball-speed", type=float, default=2.6, help="Initial speed of the ball along the ramp [m/s]."
        )
        parser.add_argument("--ramp-angle", type=float, default=18.0, help="Ramp angle above the ground [deg].")
        parser.add_argument("--iterations", type=int, default=8, help="XPBD solver iterations per substep.")
        parser.add_argument("--sim-substeps", type=int, default=16, help="Physics substeps per rendered frame.")
        parser.add_argument(
            "--disable-soft-buffer",
            action="store_true",
            help="Disable the soft buffer between the ball and the first domino.",
        )
        parser.add_argument(
            "--soft-density",
            type=float,
            default=SOFT_BUFFER_DENSITY,
            help="Soft buffer density [kg/m^3].",
        )
        parser.add_argument(
            "--soft-k-mu",
            type=float,
            default=SOFT_BUFFER_K_MU,
            help="Soft buffer shear stiffness parameter.",
        )
        parser.add_argument(
            "--soft-k-lambda",
            type=float,
            default=SOFT_BUFFER_K_LAMBDA,
            help="Soft buffer volumetric stiffness parameter.",
        )
        parser.add_argument(
            "--soft-damping",
            type=float,
            default=SOFT_BUFFER_K_DAMP,
            help="Soft buffer material damping.",
        )
        parser.add_argument(
            "--soft-surface-ke",
            type=float,
            default=0.0,
            help="Soft buffer surface triangle elastic stiffness.",
        )
        parser.add_argument(
            "--soft-surface-ka",
            type=float,
            default=0.0,
            help="Soft buffer surface triangle area stiffness.",
        )
        parser.add_argument(
            "--soft-surface-kd",
            type=float,
            default=0.0,
            help="Soft buffer surface triangle damping.",
        )
        parser.add_argument(
            "--soft-contact-margin",
            type=float,
            default=SOFT_CONTACT_MARGIN,
            help="Rigid-soft contact generation margin [m].",
        )
        parser.add_argument(
            "--soft-contact-ke",
            type=float,
            default=75.0,
            help="Rigid-soft contact stiffness [N/m].",
        )
        parser.add_argument(
            "--soft-contact-kd",
            type=float,
            default=1.0,
            help="Rigid-soft contact damping [N*s/m].",
        )
        parser.add_argument(
            "--soft-contact-kf",
            type=float,
            default=1.0e3,
            help="Rigid-soft friction force stiffness [N*s/m].",
        )
        parser.add_argument(
            "--soft-contact-mu",
            type=float,
            default=1.0,
            help="Rigid-soft contact friction coefficient.",
        )
        parser.add_argument(
            "--soft-contact-restitution",
            type=float,
            default=0.0,
            help="Rigid-soft contact restitution coefficient.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
