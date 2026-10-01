"""Developer task preparation: a hinged cover/peg and a rigid-particle pour.

All post-reset object motion is physical. Goal predicates use actual body poses.
"""
import math
import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.building import actors
from appl.envs import scene as cad
from appl.envs.native import add_box
from .scenes import ContactSequenceEnv
from .articulated_metrics import measure, rotation

NAMES=('covered_peg_assembly','granular_pour_return')


def specification(name):
    common=dict(task_id=name,robot='panda',control_hz=20,id_xy_half_width_m=.008,
        ood_xy_inner_m=.018,ood_xy_outer_m=.028,demonstrations=12,evaluation_layouts=15,position_ood_only=True)
    if name=='covered_peg_assembly':
        return dict(common,description='Open the hinged cover, retrieve the peg, turn it to align with the square hole and insert it.',
            peg_initial=[-.61,-.20,.026],peg_initial_quaternion=[math.sqrt(.5),0,0,math.sqrt(.5)],
            peg_half=[.060,.014,.014],box_center=[-.575,-.20],box_half_xy=[.125,.16],box_wall_top=.05,
            hinge=[-.45,-.20,.066],lid_handle_local=[-.17,0,.028],lid_hinge_axis=[0,1,0],lid_max_angle=1.85,
            hole_center=[-.12,.18,.10],hole_half_depth=.04,hole_half_width=.019,hole_outer_half=.065,
            target=[-.18,.18,.10],head_x_interval=[-.145,-.09],head_yz_tolerance=.009,
            minimum_axis_alignment=.985)
    if name=='granular_pour_return':
        return dict(common,description='Lift the particle-filled container, pour at least ten of twelve rigid particles into the bowl, then return the source container upright to its marked region.',
            container_initial=[-.45,-.21,.003],container_half_xy=[.035,.030],container_height=.085,wall_thickness=.005,
            container_grasp_local=[-.07,0,.052],container_handle_local=[-.07,0,.052],
            container_handle_half=[.008,.012,.025],container_bound_half_xy=[.078,.030],particle_radius=.0055,particle_count=12,
            bowl_center=[-.26,.18],bowl_inner_half_xy=[.13,.12],bowl_wall_top=.08,bowl_floor_top=.012,
            return_target=[-.46,-.28,0.],return_half_xy=[.12,.075],return_z_tolerance=.015,
            upright_cosine=.97,minimum_particles_in_bowl=10,contact_tolerance_m=.0003)
    raise ValueError(name)


@register_env('APPLSixTaskArticulated-v1',max_episode_steps=5000)
class ArticulatedSequenceEnv(ContactSequenceEnv):
    def _load_scene(self,options):
        s=self.task_spec;self.static_boxes('table',[([0,0,-.035],[1.05,.68,.035])],[.55,.48,.38,1])
        if s['task_id']=='covered_peg_assembly':
            b=self.scene.create_actor_builder();b.initial_pose=sapien.Pose(s['peg_initial'],s['peg_initial_quaternion'])
            add_box(b,[0,0,0],s['peg_half'],[.9,.30,.08,1],density=300);self.peg=b.build('assembly_peg')
            x,y=s['box_center'];hx,hy=s['box_half_xy'];h=s['box_wall_top'];w=.007
            self.static_boxes('covered_box',[([x,y,.006],[hx,hy,.006]),([x-hx-w,y,h/2],[w,hy+w,h/2]),
                ([x+hx+w,y,h/2],[w,hy+w,h/2]),([x,y-hy-w,h/2],[hx,w,h/2]),([x,y+hy+w,h/2],[hx,w,h/2])],[.45,.62,.57,1])
            b=self.scene.create_articulation_builder();root=b.create_link_builder();root.set_name('cover_mount')
            link=b.create_link_builder(root);link.set_name('cover');link.set_joint_name('cover_hinge')
            add_box(link,[-s['box_half_xy'][0],0,0],[s['box_half_xy'][0]+.008,s['box_half_xy'][1]+.008,.006],[.3,.45,.6,1],density=150)
            add_box(link,s['lid_handle_local'],[.026,.009,.018],[.65,.65,.68,1],density=150)
            link.set_joint_properties(type='revolute',limits=[[0.,s['lid_max_angle']]],
                pose_in_parent=sapien.Pose(s['hinge'],[math.sqrt(.5),0,0,math.sqrt(.5)]),
                pose_in_child=sapien.Pose(q=[math.sqrt(.5),0,0,math.sqrt(.5)]),friction=.0025,damping=.02)
            b.initial_pose=sapien.Pose();self.lid=b.build('hinged_cover',fix_root_link=True)
            x,y,z=s['hole_center'];d=s['hole_half_depth'];inner=s['hole_half_width'];outer=s['hole_outer_half']
            t=(outer-inner)/2;off=(outer+inner)/2
            self.static_boxes('insertion_fixture',[([x,y-off,z],[d,t,outer]),([x,y+off,z],[d,t,outer]),
                ([x,y,z-off],[d,inner,t]),([x,y,z+off],[d,inner,t]),
                ([x,y,(z-outer)/2],[d,outer,(z-outer)/2])],[.8,.72,.43,1])
        else:
            hx,hy=s['container_half_xy'];w=s['wall_thickness'];h=s['container_height']
            b=self.scene.create_actor_builder();b.initial_pose=sapien.Pose(s['container_initial'])
            for p,half in [([0,0,w],[hx,hy,w]),([-hx+w/2,0,h/2],[w/2,hy,h/2]),
                    ([hx-w/2,0,h/2],[w/2,hy,h/2]),([0,-hy+w/2,h/2],[hx-w,w/2,h/2]),([0,hy-w/2,h/2],[hx-w,w/2,h/2])]:
                add_box(b,p,half,[.1,.3,.8,1],density=350)
            add_box(b,[-.045,0,.052],[.025,.006,.006],[.1,.3,.8,1],density=350)
            add_box(b,s['container_handle_local'],s['container_handle_half'],[.1,.3,.8,1],density=350)
            self.container=b.build('source_container');self.particles=[]
            for i in range(s['particle_count']):
                self.particles.append(actors.build_sphere(self.scene,radius=s['particle_radius'],color=[.95,.65,.1,1],
                    name='particle_'+str(i),initial_pose=sapien.Pose([-.45,-.21,.02+.013*i])))
            x,y=s['bowl_center'];hx,hy=s['bowl_inner_half_xy'];top=s['bowl_wall_top'];floor=s['bowl_floor_top'];w=.007
            self.static_boxes('receiving_bowl',[([x,y,floor/2],[hx+w,hy+w,floor/2]),
                ([x-hx-w,y,top/2],[w,hy+w,top/2]),([x+hx+w,y,top/2],[w,hy+w,top/2]),
                ([x,y-hy-w,top/2],[hx,w,top/2]),([x,y+hy+w,top/2],[hx,w,top/2])],[.6,.66,.45,1])
            x,y,z=s['return_target'];self.static_boxes('return_marker',[([x,y,-.0005],s['return_half_xy']+[.0005])],[.25,.6,.35,1])

    def _initialize_episode(self,env_idx,options):
        if len(env_idx)!=1:raise ValueError('One environment per trial')
        s=self.task_spec;rng=np.random.default_rng(int(options.get('layout_seed',0)));condition=options.get('condition','ID')
        if condition=='ID':offset=rng.uniform(-s['id_xy_half_width_m'],s['id_xy_half_width_m'],2)
        elif condition=='OOD':
            offset=rng.uniform(-s['ood_xy_outer_m'],s['ood_xy_outer_m'],2);axis=int(rng.integers(2))
            offset[axis]=rng.choice([-1,1])*rng.uniform(s['ood_xy_inner_m'],s['ood_xy_outer_m'])
        else:raise ValueError(condition)
        self.agent.reset(np.asarray(cad.REST_QPOS));self.agent.robot.set_pose(sapien.Pose(cad.ROBOT_BASE))
        def place(actor,p,q=(1,0,0,0)):
            actor.set_pose(sapien.Pose(p,q));actor.set_linear_velocity(torch.zeros((1,3),device=self.device))
            actor.set_angular_velocity(torch.zeros((1,3),device=self.device))
        if s['task_id']=='covered_peg_assembly':
            p=np.asarray(s['peg_initial']);p[:2]+=offset;place(self.peg,p,s['peg_initial_quaternion'])
            self.lid.set_qpos(torch.zeros((1,1),device=self.device));self.lid.set_qvel(torch.zeros((1,1),device=self.device))
        else:
            p=np.asarray(s['container_initial']);p[:2]+=offset;place(self.container,p)
            for i,actor in enumerate(self.particles):
                place(actor,p+[(i%3-1)*.014,((i//3)%2-.5)*.022,.016+(i//6)*.014])

    def _get_obs_extra(self,info):
        s=self.task_spec;extra=dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        def fixed(p):return torch.tensor([list(p)+[1,0,0,0]],dtype=torch.float32,device=self.device)
        if s['task_id']=='covered_peg_assembly':
            extra.update(peg_pose=self.peg.pose.raw_pose,lid_pose=self.lid.links[1].pose.raw_pose,
                lid_position=self.lid.get_qpos(),lid_velocity=self.lid.get_qvel(),
                hole_pose=fixed(s['hole_center']),target_pose=fixed(s['target']))
        else:
            extra.update(container_pose=self.container.pose.raw_pose,
                particle_positions=torch.cat([p.pose.p for p in self.particles],dim=1),
                target_pose=fixed(s['return_target']),bowl_pose=fixed(s['bowl_center']+[s['bowl_floor_top']]))
        return extra

    def evaluate(self):
        s=self.task_spec
        if s['task_id']=='covered_peg_assembly':state=dict(peg_pose=self.peg.pose.raw_pose[0].cpu().numpy().tolist())
        else:state=dict(container_pose=self.container.pose.raw_pose[0].cpu().numpy().tolist(),
            particle_positions=torch.cat([p.pose.p for p in self.particles],dim=1)[0].cpu().numpy().tolist())
        return {k:torch.tensor([v],device=self.device) for k,v in measure(state,s).items()}


def make(spec):
    import gymnasium as gym
    return gym.make('APPLSixTaskArticulated-v1',task_spec=spec,num_envs=1,obs_mode='rgb+state_dict',
        control_mode='pd_joint_pos',sim_backend='physx_cpu',render_backend='cuda:0',reward_mode='none',max_episode_steps=5000)
