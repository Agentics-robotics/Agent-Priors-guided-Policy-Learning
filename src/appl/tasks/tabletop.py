"""Fixed native physics for task preparation and paired evaluation."""
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from ..envs import scene
from ..envs.native import add_box
from .tabletop_contract import layout,contract,measure


@register_env('APPLTabletopSequence-v1',max_episode_steps=5000)
class TabletopSequenceEnv(BaseEnv):
    SUPPORTED_ROBOTS=['panda']

    def __init__(self,*args,task_spec,**kwargs):
        self.task_spec=task_spec;self.completion_contract=contract(task_spec)
        super().__init__(*args,robot_uids='panda',**kwargs)

    @property
    def _default_sensor_configs(self):
        return [CameraConfig('front',sapien_utils.look_at([-.65,-.85,.82],[-.26,0,.08]),128,128,1.05,.01,5.)]

    @property
    def _default_human_render_camera_configs(self):return []

    def _load_agent(self,options):super()._load_agent(options,sapien.Pose(scene.ROBOT_BASE))

    def _load_scene(self,options):
        table=self.scene.create_actor_builder();table.initial_pose=sapien.Pose()
        add_box(table,[0,0,-.035],[.95,.65,.035],[.55,.48,.38,1]);table.build_static('table')
        for index,box in enumerate(self.task_spec['fixtures']):
            b=self.scene.create_actor_builder();b.initial_pose=sapien.Pose()
            add_box(b,box['position'],box['half'],[.65,.58,.43,1]);b.build_static('fixture_'+str(index))
        for name,color in [('red',[.85,.08,.06,1]),('blue',[.05,.2,.9,1])]:
            actor=actors.build_cube(self.scene,half_size=self.task_spec['object_half_m'],color=color,
                name=name+'_block',initial_pose=sapien.Pose(self.task_spec['initial'][name]))
            setattr(self,name,actor)
            goal=self.task_spec['goals'][name];pad=self.scene.create_actor_builder();pad.initial_pose=sapien.Pose()
            pad.add_box_visual(pose=sapien.Pose([goal[0],goal[1],goal[2]-.019]),
                half_size=[*self.task_spec['goal_half_xy_m'],.001],
                material=sapien.render.RenderMaterial(base_color=[*color[:3],.45]))
            pad.build_static(name+'_goal')

    def _initialize_episode(self,env_idx,options):
        if len(env_idx)!=1:raise ValueError('One environment per trial')
        self.initial=layout(self.task_spec,int(options.get('layout_seed',0)),options.get('condition','ID'))
        self.agent.reset(np.asarray(scene.REST_QPOS));self.agent.robot.set_pose(sapien.Pose(scene.ROBOT_BASE))
        for name in ('red','blue'):
            actor=getattr(self,name);actor.set_pose(sapien.Pose(self.initial[name]))
            actor.set_linear_velocity(torch.zeros((1,3),device=self.device))
            actor.set_angular_velocity(torch.zeros((1,3),device=self.device))

    def _get_obs_extra(self,info):
        return dict(tcp_pose=self.agent.tcp_pose.raw_pose,red_pose=self.red.pose.raw_pose,blue_pose=self.blue.pose.raw_pose,
            drawer_position=torch.zeros((1,1),device=self.device),drawer_velocity=torch.zeros((1,1),device=self.device),
            red_goal=torch.tensor([self.task_spec['goals']['red']],device=self.device),
            blue_goal=torch.tensor([self.task_spec['goals']['blue']],device=self.device))

    def evaluate(self):
        state={name+'_pose':getattr(self,name).pose.raw_pose[0].cpu().numpy().tolist() for name in ('red','blue')}
        return {k:torch.tensor([v],device=self.device) for k,v in measure(state,self.completion_contract).items()}

    def compute_dense_reward(self,*args,**kwargs):raise NotImplementedError('Demonstration learning only')


def make(spec,max_steps=5000):
    import gymnasium as gym
    return gym.make('APPLTabletopSequence-v1',task_spec=spec,max_episode_steps=max_steps,num_envs=1,
        obs_mode='rgb+state_dict',control_mode='pd_joint_pos',sim_backend='physx_cpu',render_backend='cuda:0',reward_mode='none')


def reset(env,seed,condition):
    from ..envs.adapter import state_from_obs
    obs,_=env.reset(seed=seed,options=dict(layout_seed=seed,condition=condition))
    return obs,state_from_obs(obs)
