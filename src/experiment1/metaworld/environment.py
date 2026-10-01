"""Audited native Task resets with a shared deployment observation schema."""
from __future__ import annotations
import pickle
import importlib
from pathlib import Path
import mujoco
import numpy as np
from .snapshot import snapshot_initial_state
from .geometry import GeometryEnv, CanonicalExpert
from .tasks import NATIVE_IDS,SOURCE_FILES

class TaskEnv:
    def __init__(self,task,render_mode=None):
        self.task=task
        if task in ('drawer','door'):
            self.geometry=GeometryEnv(task,render_mode)
            self.native=self.geometry.native
        else:
            from metaworld.env_dict import ALL_V3_ENVIRONMENTS
            self.geometry=None
            self.native=ALL_V3_ENVIRONMENTS[NATIVE_IDS[task]](render_mode=render_mode,camera_name='corner2',width=480,height=480)
            self.spec=mujoco.MjSpec.from_file(self.native.model_name)
            self.original_body_pos=self.native.model.body_pos.copy()
            self.original_site_pos=self.native.model.site_pos.copy()
            self.original_eq_data=self.native.model.eq_data.copy()
        self.native.max_path_length=500
        self.steps=0

    @property
    def model(self):return self.native.model
    @property
    def data(self):return self.native.data

    def pack(self,obs):
        if self.task=='pick-place-wall':
            wall=self.data.body('wall').xpos.copy()
            return np.r_[obs,wall,[.12,.01,.06]]
        if self.task=='assembly':return np.r_[obs,self.data.body('RoundNut').xpos]
        if self.task=='peg-insert-side':return np.r_[obs,self.data.site('pegHead').xpos]
        return obs.copy()

    def reset(self,record):
        self.steps=0
        if self.geometry:
            rec=dict(record,task_name=self.task)
            obs,info=self.geometry.reset(rec)
            return obs,info
        e=self.native
        vec=np.asarray(record['task_params']['rand_vec'],np.float64)
        low=e._random_reset_space.low.copy();high=e._random_reset_space.high.copy()
        if self.task=='assembly':low[0]=-.08;high[0]=.08
        assert np.all(vec>=low-1e-9) and np.all(vec<=high+1e-9), 'Uncalibrated native reset vector'
        assert np.linalg.norm(vec[:2]-vec[3:5]) >= (.15 if self.task=='pick-place-wall' else .1)
        # Native reset moves these fixed bodies. Compile at that same position
        # first so the static collision BVH agrees with native model.body_pos.
        if self.task in ('assembly','peg-insert-side'):
            spec=self.spec.copy()
            spec.body('peg' if self.task=='assembly' else 'box').pos=vec[3:]-(np.array([0.,0.,.05]) if self.task=='assembly' else 0.)
            model=spec.compile();data=mujoco.MjData(model)
            e.mujoco_renderer.close();e.model,e.data=model,data
            from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
            e.mujoco_renderer=MujocoRenderer(model,data,width=480,height=480,camera_name='corner2')
            e.reset_mocap_welds()
        else:
            e.model.body_pos[:]=self.original_body_pos
            e.model.site_pos[:]=self.original_site_pos
            e.model.eq_data[:]=self.original_eq_data
        mujoco.mj_resetData(e.model,e.data);mujoco.mj_forward(e.model,e.data)
        e._did_see_sim_exception=False;e._last_stable_obs=None
        e._prev_obs=e._get_curr_obs_combined_no_goal().copy()
        from metaworld.types import Task
        e.set_task(Task(NATIVE_IDS[self.task],pickle.dumps(dict(env_cls=type(e),rand_vec=vec,partially_observable=False))))
        e.seed(int(record['seed']))
        obs,_=e.reset()
        _,info=e.evaluate_state(obs.copy(),np.zeros(4,np.float32))
        info=dict(info,valid_joint=True)
        assert not info['success'],'Reset is already successful'
        assert np.isfinite(obs).all()
        np.testing.assert_array_equal(obs[:18],obs[18:36])
        np.testing.assert_allclose(obs[36:39],e._target_pos,atol=1e-9,rtol=0)
        return self.pack(obs),info

    def step(self,action):
        action=np.asarray(action,np.float32)
        assert action.shape==(4,) and np.isfinite(action).all() and np.max(np.abs(action))<=1.
        if self.steps>=500:raise RuntimeError('Episode needs reset')
        self.steps+=1
        if self.geometry:return self.geometry.step(action)
        obs,reward,terminated,truncated,info=self.native.step(action.copy())
        if self.native._did_see_sim_exception or not np.isfinite(obs).all() or not np.isfinite(self.data.qpos).all():raise RuntimeError('MuJoCo numerical failure')
        info=dict(info,valid_joint=True)
        return self.pack(obs),reward,bool(terminated),bool(truncated or self.steps>=500),info

    def snapshot(self,obs):
        if self.geometry:return self.geometry.snapshot(obs)
        return snapshot_initial_state(self.native,obs)

    def close(self):self.native.close()

class NativeExpert:
    def __init__(self,env):
        self.env=env
        name,cls={
          'pick-place-wall':('sawyer_pick_place_wall_v3_policy','SawyerPickPlaceWallV3Policy'),
          'assembly':('sawyer_assembly_v3_policy','SawyerAssemblyV3Policy'),
          'peg-insert-side':('sawyer_peg_insertion_side_v3_policy','SawyerPegInsertionSideV3Policy'),
          'stick-push':('sawyer_stick_push_v3_policy','SawyerStickPushV3Policy'),
        }[env.task]
        self.policy=getattr(importlib.import_module('metaworld.policies.'+name),cls)()
    def action(self,obs):return np.clip(self.policy.get_action(obs[:39].copy()),-1.,1.).astype(np.float32)

def make_expert(env):return CanonicalExpert(env.geometry) if env.geometry else NativeExpert(env)

def observation_schema(task):
    fields=[]
    for name,offset in [('current',0),('previous',18)]:
        for field,s,e,units in [('hand_xyz',0,3,'m'),('gripper',3,4,'unitless aperture / .1 m'),('object_xyz',4,7,'m'),('object_quaternion',7,11,'unit quaternion'),('object2_xyz',11,14,'m or absent zeros'),('object2_quaternion',14,18,'unit quaternion or absent zeros')]:
            fields.append(dict(name=name+'.'+field,slice=[offset+s,offset+e],units=units))
    fields.append(dict(name='goal_xyz',slice=[36,39],units='m'))
    if task in ('drawer','door'):fields.append(dict(name='cabinet_sin_cos_yaw',slice=[39,41],units='unitless'))
    if task=='pick-place-wall':
        fields.extend([dict(name='wall_center_xyz',slice=[39,42],units='m'),dict(name='wall_half_size_xyz',slice=[42,45],units='m')])
    if task in ('assembly','peg-insert-side'):fields.append(dict(name='ring_center_xyz' if task=='assembly' else 'peg_head_xyz',slice=[39,42],units='m'))
    return dict(version='round3-common-observation-v1',raw_dim=41 if task in ('drawer','door') else 45 if task=='pick-place-wall' else 42 if task in ('assembly','peg-insert-side') else 39,fields=fields,quaternion_order='wxyz' if task=='assembly' else 'xyzw',object2_present=task=='stick-push',frame='world; native current18+previous18+goal3',policy='state-input',all_methods_receive_identical_raw_fields=True,runtime_oracles_provided=False,extra_physical_pose_channels='ring body center for assembly; pegHead site for peg-insert-side; same declared channel for B0 and all candidates',object_point={'assembly':'RoundNut-8 site on nut handle; native quaternion is RoundNut body wxyz','peg-insert-side':'pegGrasp site','stick-push':'stick body COM; object2 insertion site+[0,.09,0]','pick-place-wall':'objGeom center','drawer':'yaw-corrected drawer handle','door':'handle geom center'}[task],native_history_reset='current repeated as previous',dp_history='two whole observations, initial repeated')
