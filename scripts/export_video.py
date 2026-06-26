"""Export the Newton experiment as an MP4 video."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image

import newton.examples
from src.main import Example


def _set_camera_from_target(viewer, target: np.ndarray, distance: float, pitch: float, yaw: float) -> None:
    camera = getattr(viewer, "camera", None)
    if camera is None:
        return

    camera.pitch = pitch
    camera.yaw = yaw
    front = camera.get_front()
    viewer.set_camera(
        wp.vec3(
            float(target[0] - front.x * distance),
            float(target[1] - front.y * distance),
            float(target[2] - front.z * distance),
        ),
        pitch=pitch,
        yaw=yaw,
    )
    camera.set_pivot(target)
    if hasattr(viewer, "_camera_dirty"):
        viewer._camera_dirty = True


def _frame_camera_around_model(viewer, state, padding: float) -> None:
    camera = getattr(viewer, "camera", None)
    if camera is None:
        return

    min_bounds = np.array([float("inf")] * 3, dtype=np.float32)
    max_bounds = np.array([float("-inf")] * 3, dtype=np.float32)
    found_objects = False

    if getattr(state, "body_q", None) is not None:
        body_q = state.body_q.numpy()
        if len(body_q) > 0:
            positions = body_q[:, :3]
            min_bounds = np.minimum(min_bounds, positions.min(axis=0))
            max_bounds = np.maximum(max_bounds, positions.max(axis=0))
            found_objects = True

    if getattr(state, "particle_q", None) is not None:
        particle_q = state.particle_q.numpy()
        if len(particle_q) > 0:
            min_bounds = np.minimum(min_bounds, particle_q.min(axis=0))
            max_bounds = np.maximum(max_bounds, particle_q.max(axis=0))
            found_objects = True

    if not found_objects:
        return

    center = 0.5 * (min_bounds + max_bounds)
    size = max_bounds - min_bounds
    max_extent = max(float(np.max(size)), 1.0)
    fov_rad = np.radians(float(getattr(camera, "fov", 60.0)))
    distance = max_extent / (2.0 * np.tan(fov_rad * 0.5)) * padding
    front = camera.get_front()

    camera.pos = type(camera.pos)(
        float(center[0] - front.x * distance),
        float(center[1] - front.y * distance),
        float(center[2] - front.z * distance),
    )
    camera.set_pivot(center)
    if hasattr(viewer, "_camera_dirty"):
        viewer._camera_dirty = True


def _set_action_camera(viewer, example: Example, args) -> None:
    camera = getattr(viewer, "camera", None)
    if camera is None:
        return

    positions = [example.domino_positions]
    if len(example.pyramid_positions) > 0:
        positions.append(example.pyramid_positions)
    scene_positions = np.vstack(positions) if positions else np.empty((0, 2), dtype=np.float32)

    if len(scene_positions) == 0:
        return

    mins = scene_positions.min(axis=0)
    maxs = scene_positions.max(axis=0)
    center_xy = 0.5 * (mins + maxs)
    size_xy = maxs - mins
    span = max(float(np.max(size_xy)), 3.0)

    target = np.array([center_xy[0], center_xy[1], 0.45], dtype=np.float32)
    if args.camera_target is not None:
        target = np.array(args.camera_target, dtype=np.float32)

    fov_rad = np.radians(float(getattr(camera, "fov", 45.0)))
    padding = 0.95 if args.camera_padding is None else args.camera_padding
    distance = span / (2.0 * np.tan(fov_rad * 0.5)) * padding
    if args.camera_distance is not None:
        distance = args.camera_distance

    _set_camera_from_target(viewer, target, distance, args.camera_pitch, args.camera_yaw)


def _set_export_camera(viewer, example: Example, args) -> None:
    if args.camera_mode == "frame":
        padding = 1.65 if args.camera_padding is None else args.camera_padding
        _frame_camera_around_model(viewer, example.state_0, padding)
        return

    if args.camera_mode == "manual":
        if args.camera_pos is not None:
            viewer.set_camera(wp.vec3(*args.camera_pos), pitch=args.camera_pitch, yaw=args.camera_yaw)
            return
        if args.camera_target is None or args.camera_distance is None:
            raise ValueError("--camera-mode manual requires either --camera-pos or both --camera-target and --camera-distance")

    _set_action_camera(viewer, example, args)


def _create_writer(video_path: Path, fps: int):
    try:
        import imageio.v2 as imageio
    except ImportError:
        return None

    video_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        video_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )


def _save_frame(frame: np.ndarray, frames_dir: Path, index: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(frames_dir / f"frame_{index:05d}.png")


def main() -> None:
    parser = Example.create_parser()
    parser.set_defaults(viewer="gl", headless=True, num_frames=2000)
    parser.add_argument("--video-output", type=Path, default=Path("outputs/newton_experiment.mp4"))
    parser.add_argument("--video-duration", type=float, default=10.0, help="Video duration [s].")
    parser.add_argument("--video-fps", type=int, default=30, help="Encoded video frame rate [frames/s].")
    parser.add_argument("--frames-dir", type=Path, default=Path("outputs/newton_experiment_frames"))
    parser.add_argument("--keep-frames", action="store_true", help="Also save PNG frames next to the MP4.")
    parser.add_argument("--record-ui", action="store_true", help="Include the viewer UI in captured frames.")
    parser.add_argument(
        "--camera-mode",
        choices=["action", "frame", "manual"],
        default="action",
        help="Camera setup: action is a tight shot around the domino/pyramid action, frame shows the whole model.",
    )
    parser.add_argument("--camera-padding", type=float, default=None, help="Camera padding scale; lower values zoom in.")
    parser.add_argument("--camera-pitch", type=float, default=-32.0, help="Camera pitch [deg].")
    parser.add_argument("--camera-yaw", type=float, default=139.0, help="Camera yaw [deg].")
    parser.add_argument("--camera-distance", type=float, default=None, help="Manual distance from camera target [m].")
    parser.add_argument("--camera-target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--camera-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    args = parser.parse_args()

    if args.viewer != "gl":
        raise ValueError("Video export uses the OpenGL viewer. Keep --viewer gl for MP4 output.")
    if args.video_duration <= 0.0:
        raise ValueError("--video-duration must be positive")
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    _set_export_camera(viewer, example, args)
    example._restore_scene()

    frame_count = int(round(args.video_duration * args.video_fps))
    writer = _create_writer(args.video_output, args.video_fps)
    save_frames = args.keep_frames or writer is None

    if writer is None:
        print("imageio/imageio-ffmpeg not available; exporting PNG frames only.")
        print("Run with: uv run --extra examples --with imageio --with imageio-ffmpeg python scripts/export_video.py")

    try:
        for frame_index in range(frame_count):
            target_time = frame_index / args.video_fps
            while example.sim_time + 0.5 * example.frame_dt < target_time:
                example.step()

            example.render()
            frame = viewer.get_frame(render_ui=args.record_ui).numpy()

            if writer is not None:
                writer.append_data(frame)
            if save_frames:
                _save_frame(frame, args.frames_dir, frame_index)

            if frame_index % max(1, args.video_fps) == 0:
                print(f"Recorded {frame_index}/{frame_count} frames")
    finally:
        if writer is not None:
            writer.close()
            print(f"Video saved to {args.video_output.resolve()}")
        viewer.close()

    if writer is None:
        print(f"PNG frames saved to {args.frames_dir.resolve()}")


if __name__ == "__main__":
    main()
