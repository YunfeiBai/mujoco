"""A dexterous hand grasps, lifts, squeezes, turns and releases a soft ball.

The hand in `hand.xml` is driven by position actuators. This script plays a
scripted sequence of hand poses that is adapted on-line to the measured state
of the deformable ball (a trilinear `flexcomp`): the hand centers itself over
the ball, descends until the palm meets it, swings the thumb behind it, wraps
the fingers around it, and later lowers it exactly as far as needed to set it
back down on the table.

Usage:
  python soft_grasp.py                     # interactive viewer
  python soft_grasp.py --record grasp.mp4  # video, needs imageio[ffmpeg]
  python soft_grasp.py --record grasp.gif  # animated GIF, needs Pillow
  python soft_grasp.py --headless          # simulate only and print a summary

For offscreen rendering on a headless machine set MUJOCO_GL=egl or osmesa.
"""

import argparse
import dataclasses
import os
import time
from typing import Callable, Optional

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(_HERE, "soft_grasp.xml")

FINGERS = ("index", "middle", "ring", "little")

# Hand poses, as targets of the position actuators (radians for hand joints).
# The mount_* slides are relative to the initial mount position and are set
# by the controller.
OPEN = {
    **{f"{f}_abd": 0.0 for f in FINGERS},
    **{f"{f}_mcp": 0.1 for f in FINGERS},
    **{f"{f}_pip": 0.1 for f in FINGERS},
    "thumb_abd": 0.0,
    "thumb_rot": 0.3,
    "thumb_mcp": 0.1,
    "thumb_ip": 0.1,
}

# Fingers spread and slightly curled, thumb half-way opposed.
PRESHAPE = {
    **OPEN,
    "index_abd": 0.2,
    "middle_abd": 0.07,
    "ring_abd": -0.07,
    "little_abd": -0.2,
    **{f"{f}_mcp": 0.35 for f in FINGERS},
    **{f"{f}_pip": 0.3 for f in FINGERS},
    "thumb_abd": 0.8,
    "thumb_rot": 1.0,
    "thumb_mcp": -0.15,
    "thumb_ip": 0.0,
}

# Thumb abducted and swung under the palm, behind the ball; fingers still open.
THUMB_BEHIND = {
    **PRESHAPE,
    "thumb_rot": 1.5,
    "thumb_mcp": 0.15,
    "thumb_ip": 0.15,
}


def grasp(squeeze: float = 1.0) -> dict[str, float]:
  """Power grasp. The targets lie inside the ball; `squeeze` sets the overshoot.

  The position actuators saturate at their force limits, so the overshoot sets
  how hard the hand squeezes the ball rather than where the fingers end up.

  Args:
    squeeze: Grasp strength, 0 is a light touch and 1 is the nominal grasp.

  Returns:
    Dictionary of actuator targets.
  """
  s = squeeze
  return {
      **PRESHAPE,
      **{f"{f}_mcp": 1.15 + 0.25 * s for f in FINGERS},
      **{f"{f}_pip": 0.8 + 0.4 * s for f in FINGERS},
      "thumb_rot": 1.5 + 0.1 * s,
      "thumb_mcp": 0.2 + 0.4 * s,
      "thumb_ip": 0.2 + 0.4 * s,
  }


def smoothstep(a: float) -> float:
  a = min(max(a, 0.0), 1.0)
  return a * a * (3 - 2 * a)


class SoftBall:
  """Measurements on the deformable ball, computed from its flex vertices."""

  def __init__(
      self, model: mujoco.MjModel, data: mujoco.MjData, name: str = "ball"
  ):
    # flexes have no named accessor in the Python bindings
    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
    if flex_id < 0:
      raise ValueError(f"no flex named '{name}'")
    start = model.flex_vertadr[flex_id]
    self._slice = slice(start, start + model.flex_vertnum[flex_id])
    self._data = data
    mujoco.mj_forward(model, data)
    self.rest_size = self.size()

  def vertices(self) -> np.ndarray:
    return self._data.flexvert_xpos[self._slice]

  def center(self) -> np.ndarray:
    return self.vertices().mean(axis=0)

  def size(self) -> np.ndarray:
    v = self.vertices()
    return v.max(axis=0) - v.min(axis=0)

  def bottom(self) -> float:
    return float(self.vertices()[:, 2].min())

  def top(self) -> float:
    return float(self.vertices()[:, 2].max())

  def squash(self) -> float:
    """Largest relative compression of the bounding box, 0 when undeformed."""
    return float(np.max(1.0 - self.size() / self.rest_size))


@dataclasses.dataclass
class Phase:
  """One segment of the scripted sequence.

  The hand moves from the targets at the start of the phase to `targets`,
  following a smoothstep profile over `duration` seconds. `on_start`, if given,
  is called once when the phase begins and returns extra targets (e.g. a mount
  position computed from the current state of the ball).
  """

  name: str
  duration: float
  targets: dict[str, float]
  on_start: Optional[Callable[[], dict[str, float]]] = None


class GraspController:
  """Plays a sequence of phases on the hand's position actuators."""

  def __init__(
      self,
      model: mujoco.MjModel,
      data: mujoco.MjData,
      ball: SoftBall,
      lift_height: float = 0.16,
      squeeze: float = 1.0,
  ):
    self.model, self.data, self.ball = model, data, ball

    # geometry of the mount: palm frame at qpos=0 and the palm's lower surface
    mujoco.mj_forward(model, data)
    palm = data.body("palm")
    self._palm_origin = palm.xpos.copy()
    palm_geom = model.geom("palm")
    self._palm_center = palm_geom.pos.copy()  # palm center in the palm frame
    self._palm_half_thickness = palm_geom.size[2]

    self._lift_height = lift_height
    self._squeeze = squeeze
    self._mount = {"mount_x": 0.0, "mount_y": 0.0, "mount_z": 0.0}
    self._grasp_z = 0.0  # mount height at which the palm touches the ball
    self.phases = self._make_phases()
    self._index = -1
    self._phase_start = 0.0
    self._from: dict[str, float] = {}
    self._to: dict[str, float] = {}
    self.ctrl = {
        **OPEN,
        "mount_x": 0.0,
        "mount_y": 0.0,
        "mount_z": 0.0,
        "wrist_yaw": 0.0,
        "wrist_pitch": 0.0,
        "wrist_roll": 0.0,
    }
    self._next_phase()

  # -- targets that depend on the state of the ball ---------------------------

  def _above_ball(self) -> dict[str, float]:
    """Center the palm over the ball, at the current height."""
    c = self.ball.center()
    self._mount["mount_x"] = c[0] - (
        self._palm_origin[0] + self._palm_center[0]
    )
    self._mount["mount_y"] = c[1] - (
        self._palm_origin[1] + self._palm_center[1]
    )
    return dict(self._mount)

  def _touch_ball(self) -> dict[str, float]:
    """Lower the palm onto the top of the ball, pressing slightly."""
    press = 0.002
    palm_bottom0 = self._palm_origin[2] - self._palm_half_thickness
    self._mount["mount_z"] = self.ball.top() - palm_bottom0 - press
    self._grasp_z = self._mount["mount_z"]
    return dict(self._mount)

  def _lift(self) -> dict[str, float]:
    self._mount["mount_z"] = self._grasp_z + self._lift_height
    return dict(self._mount)

  def _place(self) -> dict[str, float]:
    """Lower the ball by its measured height above the table."""
    press = 0.002
    z = self._mount["mount_z"] - (self.ball.bottom() - press)
    # never go much below the grasp height, e.g. if the ball was dropped
    self._mount["mount_z"] = max(z, self._grasp_z - 0.02)
    return dict(self._mount)

  def _retract(self) -> dict[str, float]:
    self._mount["mount_z"] = self._grasp_z + self._lift_height
    return dict(self._mount)

  # -- the sequence -----------------------------------------------------------

  def _make_phases(self) -> list[Phase]:
    g = grasp(self._squeeze)
    g_hard = grasp(1.6 * self._squeeze)
    g_light = grasp(0.2 * self._squeeze)
    w = lambda pose, **kw: {**pose, **kw}
    return [
        Phase("settle", 0.4, OPEN),
        Phase("preshape", 0.6, PRESHAPE, self._above_ball),
        Phase("approach", 1.0, PRESHAPE, self._touch_ball),
        Phase("thumb", 0.5, THUMB_BEHIND),
        Phase("close", 0.8, g),
        Phase("hold", 0.3, g),
        Phase("lift", 1.2, g, self._lift),
        Phase("squeeze", 0.6, g_hard),
        Phase("relax", 0.6, g),
        Phase("roll right", 0.8, w(g, wrist_roll=0.6)),
        Phase("roll left", 1.2, w(g, wrist_roll=-0.6)),
        Phase("roll back", 0.6, w(g, wrist_roll=0.0)),
        Phase("place", 1.2, g, self._place),
        # relax the grip before opening, so the ball does not spring away
        Phase(
            "loosen",
            0.6,
            {
                **g_light,
                **{
                    k: v
                    for k, v in THUMB_BEHIND.items()
                    if k.startswith("thumb")
                },
            },
        ),
        Phase("release", 0.8, PRESHAPE),
        Phase("retract", 1.0, OPEN, self._retract),
        Phase("rest", 1.0, OPEN),
    ]

  @property
  def phase(self) -> Phase:
    return self.phases[min(self._index, len(self.phases) - 1)]

  @property
  def done(self) -> bool:
    return self._index >= len(self.phases)

  def duration(self) -> float:
    return sum(p.duration for p in self.phases)

  def _next_phase(self) -> None:
    self._index += 1
    if self.done:
      return
    p = self.phases[self._index]
    self._phase_start = self.data.time
    self._from = dict(self.ctrl)
    self._to = {**self._from, **p.targets}
    if p.on_start is not None:
      self._to.update(p.on_start())

  def update(self) -> None:
    """Advance the sequence and write actuator targets into data.ctrl."""
    if not self.done:
      p = self.phases[self._index]
      a = smoothstep((self.data.time - self._phase_start) / p.duration)
      self.ctrl = {
          k: (1 - a) * self._from[k] + a * self._to[k] for k in self._to
      }
      if self.data.time - self._phase_start >= p.duration:
        self._next_phase()
    for name, value in self.ctrl.items():
      self.data.actuator(name).ctrl = value


class Recorder:
  """Renders frames from a camera that follows the hand and the ball."""

  def __init__(
      self,
      model: mujoco.MjModel,
      data: mujoco.MjData,
      ball: SoftBall,
      width: int,
      height: int,
      fps: int,
      duration: float,
  ):
    self.renderer = mujoco.Renderer(model, height, width)
    self.ball = ball
    self.fps = fps
    self.duration = duration
    self.frames: list[np.ndarray] = []
    self.camera = mujoco.MjvCamera()
    self.camera.distance = 0.52
    self.camera.elevation = -14
    self.camera.lookat[:] = self._target(data)

  def _target(self, data: mujoco.MjData) -> np.ndarray:
    return 0.5 * (self.ball.center() + data.site("palm_center").xpos) + [
        0,
        0,
        0.015,
    ]

  def maybe_capture(self, data: mujoco.MjData) -> None:
    if len(self.frames) < data.time * self.fps:
      # look between the palm and the ball, slowly orbiting around the hand
      self.camera.lookat[:] += 0.1 * (self._target(data) - self.camera.lookat)
      self.camera.azimuth = 105 + 45 * smoothstep(data.time / self.duration)
      self.renderer.update_scene(data, self.camera)
      self.frames.append(self.renderer.render().copy())

  def save(self, path: str) -> None:
    if path.lower().endswith(".gif"):
      from PIL import Image  # pylint: disable=g-import-not-at-top

      images = [Image.fromarray(f) for f in self.frames]
      images[0].save(
          path,
          save_all=True,
          append_images=images[1:],
          duration=int(1000 / self.fps),
          loop=0,
      )
    else:
      try:
        import imageio  # pylint: disable=g-import-not-at-top
      except ImportError as e:
        raise ImportError(
            "Saving videos requires `pip install imageio[ffmpeg]`; "
            "or record an animated GIF instead."
        ) from e
      imageio.mimsave(path, self.frames, fps=self.fps, quality=8)
    print(f"Saved {len(self.frames)} frames to {path}")


def main():
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument(
      "--record",
      metavar="PATH",
      help="render offscreen to a video (.mp4) or animated GIF (.gif)",
  )
  parser.add_argument(
      "--headless",
      action="store_true",
      help="simulate without rendering and print a summary",
  )
  parser.add_argument("--width", type=int, default=1280)
  parser.add_argument("--height", type=int, default=720)
  parser.add_argument("--fps", type=int, default=30)
  parser.add_argument(
      "--squeeze",
      type=float,
      default=1.0,
      help=(
          "grasp strength; 0.5-1.3 work well, much harder grips squeeze "
          "the soft ball out of the hand"
      ),
  )
  parser.add_argument(
      "--loop", action="store_true", help="repeat in the viewer"
  )
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path(SCENE)
  data = mujoco.MjData(model)
  ball = SoftBall(model, data)
  controller = GraspController(model, data, ball, squeeze=args.squeeze)

  stats = {"max_height": 0.0, "max_squash": 0.0, "max_touch": 0.0}
  touch_sensors = [f"{f}_touch" for f in ("thumb",) + FINGERS]

  def step():
    controller.update()
    mujoco.mj_step(model, data)
    if controller.phase.name in (
        "lift",
        "squeeze",
        "relax",
        "roll right",
        "roll left",
        "roll back",
    ):
      stats["max_height"] = max(stats["max_height"], ball.bottom())
    stats["max_squash"] = max(stats["max_squash"], ball.squash())
    if controller.phase.name in ("hold", "lift", "squeeze", "relax"):
      touch = max(data.sensor(s).data[0] for s in touch_sensors)
      stats["max_touch"] = max(stats["max_touch"], touch)

  if args.record or args.headless:
    recorder = None
    if args.record:
      recorder = Recorder(
          model,
          data,
          ball,
          args.width,
          args.height,
          args.fps,
          controller.duration(),
      )
    start = time.time()
    phase = None
    while not controller.done:
      if controller.phase.name != phase:
        phase = controller.phase.name
        print(
            f"t={data.time:5.2f}s  {phase:10s}  ball center"
            f" z={ball.center()[2]:.3f} m"
        )
      step()
      if recorder:
        recorder.maybe_capture(data)
    wall = time.time() - start
    print(
        f"Simulated {data.time:.1f}s in {wall:.1f}s; ball lifted"
        f" {stats['max_height']:.3f} m off the table, max compression"
        f" {100 * stats['max_squash']:.0f}%, max fingertip force"
        f" {stats['max_touch']:.1f} N."
    )
    if recorder:
      recorder.save(args.record)
    return

  from mujoco import viewer as mj_viewer  # pylint: disable=g-import-not-at-top

  frame_time = 1 / 60
  with mj_viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      frame_start = time.time()
      with viewer.lock():
        # advance the simulation by one frame, in real time if possible
        frame_end = data.time + frame_time
        while data.time < frame_end:
          if controller.done and args.loop:
            mujoco.mj_resetData(model, data)
            controller = GraspController(
                model, data, ball, squeeze=args.squeeze
            )
          step()
      viewer.sync()
      remaining = frame_time - (time.time() - frame_start)
      if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
  main()
