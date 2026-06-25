import math
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples

DOMINO_COUNT = 12
DOMINO_HALF_THICKNESS = 0.035
DOMINO_HALF_WIDTH = 0.14
DOMINO_HALF_HEIGHT = 0.40
DOMINO_SPACING = 0.36

BALL_RADIUS = 0.18
RAMP_LENGTH = 3.8
RAMP_WIDTH = 0.9
RAMP_THICKNESS = 0.12


@dataclass
class EditableBody:
    label: str
    body: int


class Example:
    def __init__(self, viewer, args):
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args

        self.domino_body_indices: list[int] = []
        self.editable_bodies: list[EditableBody] = []
        self.selected_edit_index = 0
        self.edit_position = [0.0, 0.0, 0.0]
        self.edit_rotation_deg = [0.0, 0.0, 0.0]
        self.zero_velocity_on_apply = False
        self.ball_body_index: int | None = None
        self.ramp_body_index: int | None = None

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.mu = 0.8
        builder.add_ground_plane(cfg=ground_cfg, color=wp.vec3(0.20, 0.22, 0.24), label="ground")

        self._add_ramp(builder, args.ramp_angle)
        self._add_ball(builder, args.ball_speed, args.ramp_angle)
        self._add_dominoes(builder, args.domino_count)

        builder.joint_qd = np.array(builder.body_qd).flatten().tolist()

        self.model = builder.finalize()
        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase=args.broad_phase)
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=args.iterations,
            rigid_contact_relaxation=0.85,
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
        self._load_edit_pose_from_state()

        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.9, -4.6, 2.3), pitch=-22.0, yaw=148.0)

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
        ball_cfg.density = 3500.0
        ball_cfg.mu = 0.65
        ball_cfg.restitution = 0.15
        ball_cfg.mu_rolling = 0.00002

        self.ball_body_index = builder.add_body(
            xform=wp.transform(p=wp.vec3(ball_x, 0.0, ball_z), q=wp.quat_identity()),
            label="rolling_ball",
        )
        builder.add_shape_sphere(
            self.ball_body_index,
            radius=BALL_RADIUS,
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

    def _add_dominoes(self, builder: newton.ModelBuilder, domino_count: int) -> None:
        domino_cfg = builder.default_shape_cfg.copy()
        domino_cfg.density = 250.0
        domino_cfg.mu = 0.9
        domino_cfg.restitution = 0.05

        first_x = 0.75
        for i in range(domino_count):
            x = first_x + i * DOMINO_SPACING
            body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(x, 0.0, DOMINO_HALF_HEIGHT),
                    q=wp.quat_identity(),
                ),
                label=f"domino_{i:02d}",
            )
            builder.add_shape_box(
                body,
                hx=DOMINO_HALF_THICKNESS,
                hy=DOMINO_HALF_WIDTH,
                hz=DOMINO_HALF_HEIGHT,
                cfg=domino_cfg,
                color=wp.vec3(0.92, 0.18, 0.12) if i == 0 else wp.vec3(0.95, 0.88, 0.68),
                label=f"domino_{i:02d}_shape",
            )
            self.domino_body_indices.append(body)
            self.editable_bodies.append(EditableBody(f"Domino {i:02d}", body))

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

        tilted_count = 0
        for body in self.domino_body_indices:
            q = wp.quat(*body_q[body][3:7])
            local_up = np.array(wp.quat_to_matrix(q), dtype=np.float32).reshape(3, 3)[:, 2]
            if abs(float(local_up[2])) < 0.85:
                tilted_count += 1

        if tilted_count < max(2, len(self.domino_body_indices) // 4):
            raise ValueError(f"Only {tilted_count} dominoes tilted; expected a visible chain reaction")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_broad_phase_arg(parser)
        parser.set_defaults(broad_phase="sap", num_frames=320)
        parser.add_argument("--domino-count", type=int, default=DOMINO_COUNT, help="Number of dominoes in the row.")
        parser.add_argument(
            "--ball-speed", type=float, default=2.4, help="Initial speed of the ball along the ramp [m/s]."
        )
        parser.add_argument("--ramp-angle", type=float, default=18.0, help="Ramp angle above the ground [deg].")
        parser.add_argument("--iterations", type=int, default=8, help="XPBD solver iterations per substep.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
