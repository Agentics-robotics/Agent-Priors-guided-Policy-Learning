"""Actual contact-driven ManiSkill drawer exchange task, with a Panda arm."""

import numpy as np
import sapien
import sapien.physx as physx
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from . import scene as cad


def add_box(builder, position, half, color, density=300):
    material = physx.PhysxMaterial(0.8, 0.6, 0.0)
    builder.add_box_collision(
        pose=sapien.Pose(position), half_size=half, material=material, density=density
    )
    builder.add_box_visual(
        pose=sapien.Pose(position),
        half_size=half,
        material=sapien.render.RenderMaterial(base_color=color),
    )


@register_env("APPLDrawerExchange-v2", max_episode_steps=1500)
class DrawerExchangeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="panda", **kwargs)

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                "front",
                sapien_utils.look_at(cad.CAMERA_EYE, cad.CAMERA_TARGET),
                128,
                128,
                1.05,
                0.01,
                5.0,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        return []

    def _load_agent(self, options):
        super()._load_agent(options, sapien.Pose(cad.ROBOT_BASE))

    def _load_scene(self, options):
        table = self.scene.create_actor_builder()
        table.initial_pose = sapien.Pose()
        add_box(table, [0, 0, -0.035], [0.95, 0.65, 0.035], [0.55, 0.48, 0.38, 1])
        table.build_static("table")
        cabinet = self.scene.create_actor_builder()
        cabinet.initial_pose = sapien.Pose()
        for p, h in cad.CABINET_BOXES:
            add_box(cabinet, p, h, [0.48, 0.55, 0.62, 1])
        cabinet.build_static("cabinet_shell")
        builder = self.scene.create_articulation_builder()
        root = builder.create_link_builder()
        root.set_name("drawer_mount")
        drawer = builder.create_link_builder(root)
        drawer.set_name("drawer_tray")
        drawer.set_joint_name("drawer_slide")
        for p, h in cad.DRAWER_BOXES:
            add_box(drawer, p, h, [0.72, 0.58, 0.38, 1], density=180)
        # Revolve both joint frames by pi around z so positive joint motion is -x.
        drawer.set_joint_properties(
            type="prismatic",
            limits=[[0.0, cad.TRAVEL]],
            pose_in_parent=sapien.Pose(cad.DRAWER_ORIGIN, [0, 0, 0, 1]),
            pose_in_child=sapien.Pose(q=[0, 0, 0, 1]),
            friction=0.15,
            damping=1.0,
        )
        builder.initial_pose = sapien.Pose()
        self.drawer = builder.build("drawer", fix_root_link=True)
        self.red = actors.build_cube(
            self.scene,
            half_size=cad.OBJECT_HALF,
            color=[0.85, 0.08, 0.06, 1],
            name="red_block",
            initial_pose=sapien.Pose([0.1, -0.06, 0.063]),
        )
        self.blue = actors.build_cube(
            self.scene,
            half_size=cad.OBJECT_HALF,
            color=[0.05, 0.2, 0.9, 1],
            name="blue_block",
            initial_pose=sapien.Pose([-0.16, 0.3, 0.02]),
        )
        pad = self.scene.create_actor_builder()
        pad.initial_pose = sapien.Pose()
        pad.add_box_visual(
            pose=sapien.Pose([*cad.OUTSIDE_GOAL[:2], 0.001]),
            half_size=[0.06, 0.06, 0.001],
            material=sapien.render.RenderMaterial(base_color=[0.3, 0.75, 0.35, 1]),
        )
        pad.build_static("outside_pad")

    def _initialize_episode(self, env_idx, options):
        if len(env_idx) != 1:
            raise ValueError(
                "Pilot uses one explicitly controlled environment per collector"
            )
        self.initial = cad.initial_state(
            int(options.get("layout_seed", 0)), options.get("variant", "standard")
        )
        offsets = options.get("object_xy_offsets", [0,0,0,0])
        if len(offsets) != 4 or not np.isfinite(offsets).all() or max(abs(float(x)) for x in offsets) > .08:
            raise ValueError("Invalid fixed scene reset offsets")
        self.initial["red"][:2] = (np.asarray(self.initial["red"][:2])+offsets[:2]).tolist()
        self.initial["blue"][:2] = (np.asarray(self.initial["blue"][:2])+offsets[2:]).tolist()
        self.agent.reset(np.array(cad.REST_QPOS))
        self.agent.robot.set_pose(sapien.Pose(cad.ROBOT_BASE))
        self.drawer.set_qpos(torch.tensor([[0.0]], device=self.device))
        self.drawer.set_qvel(torch.tensor([[0.0]], device=self.device))
        self.red.set_pose(sapien.Pose(self.initial["red"]))
        self.blue.set_pose(sapien.Pose(self.initial["blue"]))
        for actor in (self.red, self.blue):
            actor.set_linear_velocity(torch.zeros((1, 3), device=self.device))
            actor.set_angular_velocity(torch.zeros((1, 3), device=self.device))

    def _get_obs_extra(self, info):
        # Declared state-estimation channels for this simulation pilot, not a
        # claim of an installed real-robot perception system. No success flags.
        return dict(
            tcp_pose=self.agent.tcp_pose.raw_pose,
            red_pose=self.red.pose.raw_pose,
            blue_pose=self.blue.pose.raw_pose,
            drawer_position=self.drawer.get_qpos(),
            drawer_velocity=self.drawer.get_qvel(),
        )

    def evaluate(self):
        q = float(self.drawer.get_qpos()[0, 0])
        red = self.red.pose.p[0].cpu().numpy()
        blue = self.blue.pose.p[0].cpu().numpy()
        predicates = cad.task_predicates(
            q,
            red,
            blue,
            float(torch.linalg.norm(self.red.linear_velocity[0])),
            float(torch.linalg.norm(self.blue.linear_velocity[0])),
        )
        return {
            k: torch.tensor([v], device=self.device)
            for k, v in dict(predicates, success=all(predicates.values())).items()
        }

    def compute_dense_reward(self, *args, **kwargs):
        raise NotImplementedError("Pilot is demonstration based; no reward shaping")


def make_environment():
    import gymnasium as gym

    return gym.make(
        "APPLDrawerExchange-v2",
        num_envs=1,
        obs_mode="rgb+state_dict",
        control_mode="pd_joint_pos",
        sim_backend="physx_cpu",
        render_backend="cuda:0",
        reward_mode="none",
    )
