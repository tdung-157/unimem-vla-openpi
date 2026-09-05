#!/usr/bin/env python3
"""ROS2 deployment node for a UniMem policy on the Astribot bimanual robot.

Same client-side event memory as the MOTION2 node (both share
``custom_unimem/unimem_client.py``); what differs is the robot interface — Astribot
publishes and accepts per-body-part ``astribot_msgs`` rather than one aggregate
``JointState``, and its cameras arrive either raw (sim) or JPEG-compressed (real).

State / action layout (16 dims, grippers TRAILING the arms):

    [7 left-arm joints, 7 right-arm joints, left_gripper, right_gripper]

This ordering was verified over all 186 episodes of ``astri_coffee_making_v21``: dims
14/15 are the grippers (bimodal 0/100 %, |v| > pi in every episode) while dim 7 is a
plain radian arm joint, cross-checked against per-frame ``task_index``. Some older
``meta/info.json`` files name an interleaved layout with grippers at 7/15 — that naming
was wrong. This order drives both ``_assemble_state`` and ``_publish_action``, so it must
stay in sync with the delta-action mask in ``data_configs.py``.

Memory mode MUST match the trained checkpoint:

    pi05_astribot_unimem_keyframe_*  ->  memory_mode:=text_keyframe   (default)
    pi05_astribot_unimem_event_*     ->  memory_mode:=text
    pi05_astribot_unimem_video       ->  memory_mode:=video
    a non-UniMem checkpoint          ->  memory_mode:=none

Run it under the ROS2 Python (not ``uv``); ``run_deploy_astribot.sh`` handles that, the
workspace sourcing and the sim/real camera topic switch. Keyboard: 'r' new rollout,
'c' next task, 'e' show event history.
"""

import pathlib
import sys
import threading

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    from astribot_msgs.msg import RobotJointController
    from astribot_msgs.msg import RobotJointState
    from cv_bridge import CvBridge
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSDurabilityPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import QoSReliabilityPolicy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        f"This node requires a sourced ROS2 environment with astribot_msgs available.\nOriginal import error: {e}"
    )

import cv2

from custom_unimem import deploy_common
from custom_unimem import event_vocab
from custom_unimem import robot_paths
from custom_unimem import unimem_client

JOINT_PARTS: list[tuple[str, int]] = [
    ("astribot_arm_left", 7),
    ("astribot_arm_right", 7),
    ("astribot_gripper_left", 1),
    ("astribot_gripper_right", 1),
]
PART_NAMES = [part for part, _ in JOINT_PARTS]
GRIPPER_PARTS = {"astribot_gripper_left", "astribot_gripper_right"}
STATE_DIM = sum(dof for _, dof in JOINT_PARTS)  # 16
CAM_KEYS = ("head", "left_wrist", "right_wrist")
OBS_KEY_BY_CAM = {
    "head": "observation/image",
    "left_wrist": "observation/left_wrist_image",
    "right_wrist": "observation/right_wrist_image",
}

PATHS = robot_paths.ASTRIBOT

# Fixed-stride video baseline only: must match VIDEO_NUM_FRAMES / VIDEO_FRAME_STRIDE_SEC
# in robot_configs.py — 4 frames at a 2 s stride, which on a 30 fps stream is every 60th
# frame (upstream's xarm_mem7_video layout). A mismatch silently feeds the model a clip
# with different temporal spacing than it trained on.
DEFAULT_VIDEO_NUM_FRAMES = 4
DEFAULT_VIDEO_STRIDE_FRAMES = 60
DEFAULT_KEYFRAME_NUM_FRAMES = 4

QOS_CMD_PUB = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=10,
)
QOS_STATE_SUB = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    depth=15,
)


class AstribotUniMemDeployNode(Node):
    def __init__(self) -> None:
        super().__init__("astribot_unimem_deploy")

        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8000)
        self.declare_parameter("prompt", PATHS.prompt)
        self.declare_parameter("tasks_file", str(pathlib.Path(__file__).resolve().parent / "tasks_astribot.yaml"))
        self.declare_parameter("memory_mode", "text_keyframe")
        # Steps of each chunk executed before re-querying; also the event-detection period.
        self.declare_parameter("action_horizon", 25)
        self.declare_parameter("control_hz", 30.0)
        self.declare_parameter("event_confidence_threshold", unimem_client.DEFAULT_EVENT_CONFIDENCE_THRESHOLD)
        self.declare_parameter("video_num_frames", DEFAULT_VIDEO_NUM_FRAMES)
        self.declare_parameter("video_stride_frames", DEFAULT_VIDEO_STRIDE_FRAMES)
        self.declare_parameter("gripper_min_mm", 0.0)
        self.declare_parameter("gripper_max_mm", 100.0)
        self.declare_parameter("camera_transport", "compressed")
        self.declare_parameter("head_image_topic", "/astribot_camera/head_rgbd/color_compress/compressed")
        self.declare_parameter("left_wrist_image_topic", "/astribot_camera/left_wrist_rgbd/color_compress/compressed")
        self.declare_parameter("right_wrist_image_topic", "/astribot_camera/right_wrist_rgbd/color_compress/compressed")
        self.declare_parameter("state_topic_template", "/{part}/joint_space_states")
        self.declare_parameter("command_topic_template", "/{part}/joint_space_command")
        self.declare_parameter("prompt_topic", "/vla/prompt")
        self.declare_parameter("event_history_topic", "/vla/event_history")

        p = self.get_parameter
        host = str(p("host").value)
        port = int(p("port").value)
        mode = str(p("memory_mode").value)
        control_hz = float(p("control_hz").value)
        self._gripper_min = float(p("gripper_min_mm").value)
        self._gripper_max = float(p("gripper_max_mm").value)
        transport = str(p("camera_transport").value).lower()
        self._use_compressed = transport in ("compressed", "compress", "jpeg", "real")

        self._prompt_lock = threading.Lock()
        self._tasks = deploy_common.load_tasks(str(p("tasks_file").value), self.get_logger().warning)
        self._task_index = 0
        self._prompt = self._tasks[0] if self._tasks else str(p("prompt").value)
        self._check_prompt(self._prompt)

        vocab = event_vocab.get_vocab(PATHS.vocab)
        self.get_logger().info(f"Connecting to policy server ws://{host}:{port} (memory_mode={mode}) ...")
        self._policy = unimem_client.UniMemPolicyClient(
            host=host,
            port=port,
            prompt=self._prompt,
            event_phrases=vocab.phrases,
            memory_mode=mode,
            action_horizon=int(p("action_horizon").value),
            num_frames=int(p("video_num_frames").value),
            frame_stride_frames=int(p("video_stride_frames").value),
            event_confidence_threshold=float(p("event_confidence_threshold").value),
            log=self.get_logger().info,
        )
        self.get_logger().info(f"Server metadata: {self._policy.server_metadata}")

        self._bridge = CvBridge()
        self._state_lock = threading.Lock()
        self._part_state: dict[str, np.ndarray | None] = dict.fromkeys(PART_NAMES)
        self._img_lock = threading.Lock()
        self._images: dict[str, np.ndarray | None] = dict.fromkeys(CAM_KEYS)

        img_type = CompressedImage if self._use_compressed else Image
        cam_topics = {
            "head": p("head_image_topic").value,
            "left_wrist": p("left_wrist_image_topic").value,
            "right_wrist": p("right_wrist_image_topic").value,
        }
        for key, topic in cam_topics.items():
            self.create_subscription(
                img_type, topic, lambda msg, k=key: self._on_image(msg, k), qos_profile_sensor_data
            )

        state_template = str(p("state_topic_template").value)
        for part in PART_NAMES:
            self.create_subscription(
                RobotJointState,
                state_template.format(part=part),
                lambda msg, pt=part: self._on_joint_state(msg, pt),
                QOS_STATE_SUB,
            )

        command_template = str(p("command_topic_template").value)
        self._cmd_pubs = {
            part: self.create_publisher(RobotJointController, command_template.format(part=part), QOS_CMD_PUB)
            for part in PART_NAMES
        }

        self.create_subscription(String, p("prompt_topic").value, self._on_prompt, 10)
        self._history_pub = self.create_publisher(String, p("event_history_topic").value, 10)

        self.create_timer(1.0 / control_hz, self._control_tick)

        self._keys = deploy_common.KeyListener(
            {"r": self._reset_rollout, "c": self._advance_task, "e": self._print_events},
            self.get_logger().warning,
        )
        started = self._keys.start()
        if self._tasks and started:
            lines = "\n".join(f"    {i + 1:>2}. {t}" for i, t in enumerate(self._tasks))
            self.get_logger().info(f"Loaded {len(self._tasks)} tasks:\n{lines}")
        if started:
            self.get_logger().info("Keys: 'r' new rollout, 'c' next task, 'e' show event history.")

        # Frames the model sees per camera, for the banner only: the server owns the
        # keyframe cache depth, the client owns the fixed-stride stack, and the
        # single-frame modes use exactly one.
        if mode in unimem_client.KEYFRAME_MODES:
            depth = DEFAULT_KEYFRAME_NUM_FRAMES
        elif mode == "video":
            depth = int(p("video_num_frames").value)
        else:
            depth = 1
        self.get_logger().info(
            f"Ready. control_hz={control_hz}, action_horizon={p('action_horizon').value}, "
            f"memory_mode={mode}, frames={depth}, transport={transport}, prompt='{self._prompt}'. "
            "Waiting for cameras + joint states..."
        )

    def stop(self) -> None:
        self._keys.stop()

    def _check_prompt(self, prompt: str) -> None:
        """Warn when the served prompt is not the one the checkpoint was trained on.

        Every frame of training saw exactly one instruction (``robot_paths``' ``prompt``,
        injected by ``default_prompt`` — the same single-string-per-episode behaviour
        upstream gets from ``prompt_from_task=True`` on datasets whose task field is a
        per-session constant). Sending anything else puts the model somewhere it has never
        been, and it fails quietly rather than loudly, so say so.
        """
        if prompt != PATHS.prompt:
            self.get_logger().warning(
                f"prompt {prompt!r} differs from the trained prompt {PATHS.prompt!r}. "
                "The policy was conditioned on one fixed instruction for every frame; "
                "progress is supposed to reach it through phase_history, not the prompt."
            )

    # ------------------------------------------------------------------ keyboard
    def _reset_rollout(self) -> None:
        self._policy.reset()
        self.get_logger().info("--- new rollout ---")

    def _advance_task(self) -> None:
        with self._prompt_lock:
            if not self._tasks:
                return
            self._task_index = (self._task_index + 1) % len(self._tasks)
            self._prompt = self._tasks[self._task_index]
            index, total, prompt = self._task_index + 1, len(self._tasks), self._prompt
        self._policy.set_prompt(prompt)
        self._check_prompt(prompt)
        self.get_logger().info(f"[task {index}/{total}] prompt -> '{prompt}'")

    def _print_events(self) -> None:
        self.get_logger().info(f"event history: {self._policy.history_text}")
        probs = self._policy.last_event_probabilities
        if probs is not None:
            ranked = np.argsort(probs)[::-1][:3]
            self.get_logger().info("  last event-head output: " + ", ".join(f"{int(i)}:{probs[i]:.3f}" for i in ranked))

    # ------------------------------------------------------------------ callbacks
    def _on_prompt(self, msg: String) -> None:
        new_prompt = msg.data.strip()
        with self._prompt_lock:
            if new_prompt and new_prompt != self._prompt:
                self._prompt = new_prompt
                self._policy.set_prompt(new_prompt)
                self._check_prompt(new_prompt)
                self.get_logger().info(f"Prompt updated via topic: '{new_prompt}'")

    def _on_image(self, msg, key: str) -> None:
        try:
            if self._use_compressed:
                buf = np.frombuffer(msg.data, dtype=np.uint8)
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        except Exception as e:
            self.get_logger().warning(f"Failed to decode {key} image: {e}", once=True)
            return
        with self._img_lock:
            self._images[key] = rgb

    def _on_joint_state(self, msg: RobotJointState, part: str) -> None:
        with self._state_lock:
            self._part_state[part] = np.asarray(msg.position, dtype=np.float32)

    def _assemble_state(self) -> np.ndarray | None:
        parts = []
        with self._state_lock:
            for part, dof in JOINT_PARTS:
                pos = self._part_state.get(part)
                if pos is None:
                    self.get_logger().warning(f"Waiting for state on part '{part}' ...", once=True)
                    return None
                if len(pos) < dof:
                    self.get_logger().warning(f"Part '{part}' reported {len(pos)} positions, need {dof}.", once=True)
                    return None
                parts.append(pos[:dof])
        return np.concatenate(parts).astype(np.float32)

    def _snapshot_images(self) -> dict[str, np.ndarray] | None:
        with self._img_lock:
            missing = [key for key in CAM_KEYS if self._images[key] is None]
            if missing:
                self.get_logger().warning(f"Waiting for camera(s): {missing} ...", once=True)
                return None
            return {OBS_KEY_BY_CAM[key]: self._images[key] for key in CAM_KEYS}

    # ------------------------------------------------------------------ control loop
    def _control_tick(self) -> None:
        state = self._assemble_state()
        if state is None:
            return
        images = self._snapshot_images()
        if images is None:
            return

        try:
            action = self._policy.step(state=state, images=images)
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            return

        action = np.asarray(action).reshape(-1)
        if action.shape[0] < STATE_DIM:
            self.get_logger().error(f"Unexpected action dim {action.shape[0]} (< {STATE_DIM})")
            return

        self._publish_action(action)

    def _publish_action(self, action: np.ndarray) -> None:
        stamp = self.get_clock().now().to_msg()
        offset = 0
        for part, dof in JOINT_PARTS:
            segment = action[offset : offset + dof]
            offset += dof
            values = [float(v) for v in segment]
            if part in GRIPPER_PARTS:
                values = [float(np.clip(values[0], self._gripper_min, self._gripper_max))]
            if any(not np.isfinite(v) for v in values):
                self.get_logger().warning(f"Non-finite command for {part}; skipped.", once=True)
                continue
            msg = RobotJointController()
            msg.header.stamp = stamp
            msg.mode = 1
            msg.command = values
            self._cmd_pubs[part].publish(msg)

        history = String()
        history.data = self._policy.history_text
        self._history_pub.publish(history)


def main() -> None:
    rclpy.init()
    node = AstribotUniMemDeployNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
