"""Developer-owned contact environments for the two newly authorized tasks.

Task preparation is separate from API-owned policy design. No actor pose updates
occur during execution: reset initializes actors, all subsequent motion is physics.
"""
import copy
import numpy as np
import sapien
import torch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from appl.envs import scene as cad
from appl.envs.native import add_box
from appl.envs.adapter import state_from_obs
from appl.envs.evaluator import extent_wxyz

NAMES=('tool_retrieve_pack','constrained_retrieve_store')


def specification(name):
    if name in ('covered_peg_assembly','granular_pour_return'):
        from .articulated_scenes import specification as additional
        return additional(name)
    if name not in NAMES:raise ValueError(name)
    common=dict(task_id=name,robot='panda',control_hz=20,object_half_m=.02,
        id_xy_half_width_m=.008,ood_xy_inner_m=.018,ood_xy_outer_m=.028,
        demonstrations=12,evaluation_layouts=15,position_ood_only=True)
    if name=='tool_retrieve_pack':
        return dict(common,description='Use the loose L-shaped tool to retrieve a distant block, then place the block in the marked tray region.',
            object_initial=[.18,-.18,.02],tool_initial=[-.50,-.25,.018],
            tool_length=.52,tool_handle_half_y=.010,tool_half_z=.015,
            tool_hook_center=[.50,.045,0],tool_hook_half=[.020,.055,.015],
            tool_grasp_local=[.06,0,0],tool_rest=[-.47,-.37,.018],
            target=[-.34,.22,.036],target_half_xy=[.075,.075],target_z_tolerance=.012,
            tray_center=[-.34,.22],tray_half_xy=[.15,.13],tray_wall_top=.08)
    return dict(common,description='Retrieve the block from the low tunnel, put it inside the drawer, and leave the drawer closed.',
        object_initial=[-.44,-.40,.08],tunnel_center=[-.37,-.40],tunnel_half_xy=[.10,.145],
        tunnel_floor_top=.06,tunnel_clear_height=.16,tunnel_wall_thickness=.008,
        drawer_origin=list(cad.DRAWER_ORIGIN),drawer_travel=cad.TRAVEL,
        drawer_goal_local=[-.065,0,.028],drawer_closed_threshold=.025,
        cavity_half_xy=[.172,.182],object_z_interval=[.053,.074])


def initial_layout(spec,seed,condition):
    if condition not in ('ID','OOD'):raise ValueError(condition)
    rng=np.random.default_rng(seed)
    def offset():
        if condition=='ID':return rng.uniform(-spec['id_xy_half_width_m'],spec['id_xy_half_width_m'],2)
        v=rng.uniform(-spec['ood_xy_outer_m'],spec['ood_xy_outer_m'],2)
        axis=int(rng.integers(2));v[axis]=rng.choice([-1,1])*rng.uniform(spec['ood_xy_inner_m'],spec['ood_xy_outer_m'])
        return v
    result={}
    for key in ('object','tool'):
        if key+'_initial' in spec:
            p=np.asarray(spec[key+'_initial'],float);p[:2]+=offset();result[key]=p.tolist()
    return result


def measure(state,spec):
    if spec['task_id'] in ('covered_peg_assembly','granular_pour_return'):
        from .articulated_scenes import measure as additional
        return additional(state,spec)
    p=np.asarray(state['object_pose'][:3]);ext=extent_wxyz(state['object_pose'][3:],spec['object_half_m'])
    if spec['task_id']=='tool_retrieve_pack':
        g=np.asarray(spec['target'])
        inside=bool(np.all(np.abs(p[:2]-g[:2])+ext[:2]<=spec['target_half_xy']) and abs(p[2]-g[2])<spec['target_z_tolerance'])
        return dict(object_in_tray=inside,success=inside)
    q=float(state['drawer_position'][0]);center=np.asarray(spec['drawer_origin'])+[-q,0,0]
    inside=bool(np.all(np.abs(p[:2]-center[:2])+ext[:2]<=spec['cavity_half_xy']) and spec['object_z_interval'][0]<p[2]<spec['object_z_interval'][1])
    closed=q<spec['drawer_closed_threshold']
    return dict(object_inside=inside,drawer_closed=closed,success=inside and closed)


@register_env('APPLSixTaskContact-v1',max_episode_steps=5000)
class ContactSequenceEnv(BaseEnv):
    SUPPORTED_ROBOTS=['panda']

    def __init__(self,*args,task_spec,**kwargs):
        self.task_spec=copy.deepcopy(task_spec)
        super().__init__(*args,robot_uids='panda',**kwargs)

    @property
    def _default_sensor_configs(self):
        return [CameraConfig('front',sapien_utils.look_at([-.65,-.90,.85],[-.18,0,.07]),256,256,1.10,.01,5.)]

    @property
    def _default_human_render_camera_configs(self):return []

    def _load_agent(self,options):super()._load_agent(options,sapien.Pose(cad.ROBOT_BASE))

    def static_boxes(self,name,boxes,color):
        b=self.scene.create_actor_builder();b.initial_pose=sapien.Pose()
        for p,h in boxes:add_box(b,p,h,color)
        return b.build_static(name)

    def _load_scene(self,options):
        s=self.task_spec
        self.static_boxes('table',[([0,0,-.035],[1.05,.68,.035])],[.55,.48,.38,1])
        self.object=actors.build_cube(self.scene,half_size=.02,color=[.85,.08,.06,1],name='target_block',initial_pose=sapien.Pose(s['object_initial']))
        if s['task_id']=='tool_retrieve_pack':
            b=self.scene.create_actor_builder();b.initial_pose=sapien.Pose(s['tool_initial'])
            add_box(b,[s['tool_length']/2,0,0],[s['tool_length']/2,s['tool_handle_half_y'],s['tool_half_z']],[.08,.22,.85,1],density=200)
            add_box(b,s['tool_hook_center'],s['tool_hook_half'],[.08,.22,.85,1],density=200)
            self.tool=b.build('loose_l_tool')
            x,y=s['tray_center'];hx,hy=s['tray_half_xy'];top=s['tray_wall_top']
            boxes=[([x,y,.008],[hx,hy,.008]),([x-hx,y,top/2],[.008,hy,top/2]),
                ([x+hx,y,top/2],[.008,hy,top/2]),([x,y-hy,top/2],[hx,.008,top/2]),([x,y+hy,top/2],[hx,.008,top/2])]
            self.static_boxes('tray',boxes,[.58,.65,.48,1])
        else:
            x,y=s['tunnel_center'];hx,hy=s['tunnel_half_xy'];floor=s['tunnel_floor_top'];h=floor+s['tunnel_clear_height'];w=s['tunnel_wall_thickness']
            boxes=[([x,y,floor/2],[hx+.04,hy+w,floor/2]),([x,y,h+w],[hx+w,hy+w,w]),
                ([x,y-hy-w,h/2],[hx+w,w,h/2]),([x,y+hy+w,h/2],[hx+w,w,h/2])]
            self.static_boxes('low_tunnel',boxes,[.40,.46,.55,1])
            self.static_boxes('cabinet',cad.CABINET_BOXES,[.48,.55,.62,1])
            b=self.scene.create_articulation_builder();root=b.create_link_builder();root.set_name('mount')
            link=b.create_link_builder(root);link.set_name('drawer_tray');link.set_joint_name('drawer_slide')
            for p,half in cad.DRAWER_BOXES:add_box(link,p,half,[.72,.58,.38,1],density=180)
            link.set_joint_properties(type='prismatic',limits=[[0.,cad.TRAVEL]],
                pose_in_parent=sapien.Pose(cad.DRAWER_ORIGIN,[0,0,0,1]),pose_in_child=sapien.Pose(q=[0,0,0,1]),friction=.15,damping=1.)
            b.initial_pose=sapien.Pose();self.drawer=b.build('drawer',fix_root_link=True)

    def _initialize_episode(self,env_idx,options):
        if len(env_idx)!=1:raise ValueError('One physical environment per trial')
        self.initial=initial_layout(self.task_spec,int(options.get('layout_seed',0)),options.get('condition','ID'))
        self.agent.reset(np.asarray(cad.REST_QPOS));self.agent.robot.set_pose(sapien.Pose(cad.ROBOT_BASE))
        for name,p in self.initial.items():
            actor=getattr(self,name);actor.set_pose(sapien.Pose(p))
            actor.set_linear_velocity(torch.zeros((1,3),device=self.device));actor.set_angular_velocity(torch.zeros((1,3),device=self.device))
        if self.task_spec['task_id']=='constrained_retrieve_store':
            self.drawer.set_qpos(torch.zeros((1,1),device=self.device));self.drawer.set_qvel(torch.zeros((1,1),device=self.device))

    def _get_obs_extra(self,info):
        s=self.task_spec
        extra=dict(tcp_pose=self.agent.tcp.pose.raw_pose,object_pose=self.object.pose.raw_pose)
        if s['task_id']=='tool_retrieve_pack':
            extra.update(tool_pose=self.tool.pose.raw_pose,target_pose=torch.tensor([s['target']+[1,0,0,0]],device=self.device))
        else:
            q=self.drawer.get_qpos();target=torch.tensor([s['drawer_origin']],device=self.device)+torch.tensor([s['drawer_goal_local']],device=self.device)
            target[:,0]-=q[:,0]
            extra.update(drawer_position=q,drawer_velocity=self.drawer.get_qvel(),
                target_pose=torch.cat([target,torch.tensor([[1,0,0,0]],device=self.device)],dim=1))
        return extra

    def evaluate(self):
        s=dict(object_pose=self.object.pose.raw_pose[0].cpu().numpy().tolist())
        if self.task_spec['task_id']=='constrained_retrieve_store':s['drawer_position']=self.drawer.get_qpos()[0].cpu().numpy().tolist()
        return {k:torch.tensor([v],device=self.device) for k,v in measure(s,self.task_spec).items()}

    def compute_dense_reward(self,*args,**kwargs):raise NotImplementedError('Demonstration learning only')


def make(spec):
    if spec['task_id'] in ('covered_peg_assembly','granular_pour_return'):
        from .articulated_scenes import make as additional
        return additional(spec)
    import gymnasium as gym
    return gym.make('APPLSixTaskContact-v1',task_spec=spec,num_envs=1,obs_mode='rgb+state_dict',
        control_mode='pd_joint_pos',sim_backend='physx_cpu',render_backend='cuda:0',reward_mode='none',max_episode_steps=5000)


def reset(env,seed,condition='ID'):
    obs,_=env.reset(seed=seed,options=dict(layout_seed=seed,condition=condition))
    return obs,state_from_obs(obs)
