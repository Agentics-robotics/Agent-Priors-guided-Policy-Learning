from __future__ import annotations
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .snapshot import make_env, make_expert, reset_from_record, snapshot_initial_state


def rz(degrees):
    return Rotation.from_euler('z', degrees, degrees=True).as_matrix()


class GeometryEnv:
    """Only stationary cabinet root rotates. Native Cartesian control is retained."""
    def __init__(self, task, render_mode=None):
        self.task = task
        self.native = make_env(task, render_mode)
        self.model, self.data = self.native.model, self.native.data
        self.original_quat = self.model.body_quat.copy()
        self.spec = mujoco.MjSpec.from_file(self.native.model_name)
        # The stock rear retaining rail at world y=.99 intersects the rotated
        # closed drawer and pushes it open without robot contact. Move only this
        # accessory rail to y=1.15 for BOTH tasks and ALL yaw values. The table,
        # robot, gravity, mechanism dimensions and dynamics remain native.
        rear = [g for g in self.spec.body('RetainingWall').geoms if g.pos[1] > .3]
        assert len(rear) == 1
        rear[0].pos[1] = .55
        self.root = self.model.body(task).id
        self.joint = self.model.joint('goal_slidey' if task == 'drawer' else 'doorjoint').id
        self.qadr = int(self.model.jnt_qposadr[self.joint])
        self.q_closed, self.q_open = 0., float(self.model.jnt_range[self.joint, 0])
        assert self.model.jnt_limited[self.joint] and self.q_open < 0
        self.fk = mujoco.MjData(self.model)
        self.handle_geoms = []
        for i in range(self.model.ngeom):
            body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[i]))
            p = self.model.geom_pos[i]
            collides = self.model.geom_contype[i] or self.model.geom_conaffinity[i]
            if task == 'drawer':
                is_handle = body == 'drawer_link' and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE and p[1] < -.1
            else:
                is_handle = body == 'door_link' and self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER and p[0] > .3
            if is_handle and collides:
                self.handle_geoms.append(i)
        self.gripper_geoms = []
        for i in range(self.model.ngeom):
            b = int(self.model.geom_bodyid[i])
            names = []
            while b:
                names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or '')
                b = int(self.model.body_parentid[b])
            if any(n in ('leftclaw', 'rightclaw', 'leftpad', 'rightpad') for n in names):
                if self.model.geom_contype[i] or self.model.geom_conaffinity[i]:
                    self.gripper_geoms.append(i)
        assert self.handle_geoms and self.gripper_geoms

    def handle_pose(self, data=None):
        d = self.data if data is None else data
        if self.task == 'drawer':
            b = d.body('drawer_link')
            mat = b.xmat.reshape(3, 3)
            p = b.xpos + mat @ np.array([0., -.16, 0.])
        else:
            g = d.geom('handle')
            p, mat = g.xpos.copy(), g.xmat.reshape(3, 3)
        return p.copy(), Rotation.from_matrix(mat).as_quat()

    def current(self):
        cur = self.native._get_curr_obs_combined_no_goal().copy()
        cur[4:7], quat = self.handle_pose()
        reference = getattr(self, 'last_quat', np.array([0., 0., 0., 1.]))
        if np.dot(quat, reference) < 0:
            quat = -quat
        cur[7:11] = quat
        self.last_quat = quat.copy()
        return cur

    def pack(self, cur):
        return np.r_[cur, self.previous, self.goal, np.sin(np.deg2rad(self.yaw)), np.cos(np.deg2rad(self.yaw))]

    def reset(self, record):
        self.yaw = float(record['task_params']['yaw_degrees'])
        self.R = rz(self.yaw)
        # Static bodies require recompilation: mj_forward/mj_setConst alone do
        # not rebuild the static collision BVH. Never mutate a compiled yaw.
        spec = self.spec.copy()
        quat = Rotation.from_matrix(self.R).as_quat()
        spec.body(self.task).quat = quat[[3, 0, 1, 2]]
        spec.body(self.task).pos = record['task_params']['base_position']
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.fk = mujoco.MjData(self.model)
        self.native.mujoco_renderer.close()
        self.native.model, self.native.data = self.model, self.data
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        self.native.mujoco_renderer = MujocoRenderer(self.model, self.data, width=480, height=480, camera_name='corner2')
        self.native.reset_mocap_welds()
        self.native._appl_initial_body_pos = self.model.body_pos.copy()
        self.native._appl_initial_site_pos = self.model.site_pos.copy()
        self.native._appl_initial_eq_data = self.model.eq_data.copy()
        reset_from_record(self.native, record)
        self.base = self.data.body(self.task).xpos.copy()
        self.data.qpos[self.qadr] = self.q_closed
        self.data.qvel[self.model.jnt_dofadr[self.joint]] = 0.
        mujoco.mj_forward(self.model, self.data)
        self.fk.qpos[:] = self.data.qpos
        self.fk.qpos[self.qadr] = self.q_closed + .8 * (self.q_open - self.q_closed)
        mujoco.mj_forward(self.model, self.fk)
        self.goal, self.goal_quat = self.handle_pose(self.fk)
        self.native._target_pos = self.goal.copy()
        for site in self.native._target_site_config:
            self.native._set_pos_site(*site)
        mujoco.mj_forward(self.model, self.data)
        self.last_quat = np.array([0., 0., 0., 1.])
        cur = self.current()
        self.previous = cur.copy()
        self.native._prev_obs = cur.copy()
        self.streak = 0
        self.steps = 0
        self.old_progress = self.progress
        obs = self.pack(cur)
        return obs, self.diagnostics()

    @property
    def progress(self):
        return float((self.data.qpos[self.qadr] - self.q_closed) / (self.q_open - self.q_closed))

    def diagnostics(self):
        direct = []
        for c in self.data.contact[:self.data.ncon]:
            a, b = int(c.geom1), int(c.geom2)
            if (a in self.handle_geoms and b in self.gripper_geoms) or (b in self.handle_geoms and a in self.gripper_geoms):
                direct.append([a, b])
        return dict(q=float(self.data.qpos[self.qadr]), progress=self.progress,
                    success=self.streak >= 3, success_streak=self.streak,
                    distance=float(np.linalg.norm(self.native.get_endeff_pos() - self.handle_pose()[0])),
                    direct_contact=bool(direct), contact_pairs=direct,
                    valid_joint=bool(-.01 <= self.progress <= 1.01))

    def step(self, action):
        action = np.asarray(action, np.float64)
        assert action.shape == (4,) and np.isfinite(action).all() and np.max(np.abs(action)) <= 1.00000001
        if self.steps >= 500:
            raise RuntimeError('Episode needs reset')
        self.native.set_xyz_action(action[:3])
        self.native.do_simulation([action[3], -action[3]], n_frames=self.native.frame_skip)
        self.steps += 1
        self.native.curr_path_length = self.steps
        for site in self.native._target_site_config:
            self.native._set_pos_site(*site)
        mujoco.mj_forward(self.model, self.data)
        if self.native._did_see_sim_exception or not np.isfinite(self.data.qpos).all():
            raise RuntimeError('MuJoCo numerical failure')
        cur = self.current()
        obs = self.pack(cur)
        self.previous = cur.copy()
        self.native._prev_obs = cur.copy()
        p = self.progress
        self.streak = self.streak + 1 if .75 <= p <= 1.01 else 0
        reward = p - self.old_progress
        self.old_progress = p
        return obs, reward, False, self.steps == 500, self.diagnostics()

    def snapshot(self, obs):
        snap = snapshot_initial_state(self.native, obs)
        snap.update(snapshot_model_body_quat=self.model.body_quat.copy(),
                    # Original snapshot field retained for published reset compatibility.
                    snapshot_round2_previous=self.previous.copy(),
                    snapshot_goal_quat=self.goal_quat.copy(),
                    snapshot_yaw=np.asarray(self.yaw), snapshot_streak=np.asarray(self.streak))
        return snap

    def close(self):
        self.native.close()


class CanonicalExpert:
    def __init__(self, env):
        self.env = env
        self.expert = make_expert(env.task)
        self.phase = 0

    def action(self, obs):
        e = self.env
        if e.task == 'door':
            hand, handle = obs[:3], obs[4:7]
            turn = e.R @ Rotation.from_euler('z', float(e.data.qpos[e.qadr])).as_matrix()
            approach = handle + turn @ np.array([.01, .02, 0.])
            if self.phase == 0 and np.linalg.norm(hand[:2] - approach[:2]) < .035:
                self.phase = 1
            if self.phase == 1 and abs(hand[2] - approach[2]) < .025:
                self.phase = 2
            target = approach + [0., 0., .18] if self.phase == 0 else approach
            if self.phase == 2:
                target = handle + turn @ np.array([0., -.07, 0.])
            gain = 10. if self.phase < 2 else 25.
            return np.r_[np.clip(gain * (target-hand), -1, 1), 1.].astype(np.float32)
        canonical = obs[:39].copy()
        for start in (0, 18):
            for offset in (0, 4):
                s = slice(start + offset, start + offset + 3)
                canonical[s] = e.base + e.R.T @ (obs[s] - e.base)
            q = slice(start + 7, start + 11)
            canonical[q] = Rotation.from_matrix(e.R.T @ Rotation.from_quat(obs[q]).as_matrix()).as_quat()
        canonical[36:39] = e.base + e.R.T @ (obs[36:39] - e.base)
        action = self.expert.get_action(canonical.copy()).astype(np.float64)
        if e.task == 'drawer' and e.progress > .35:
            action[:3] *= .2
        action[:3] = e.R @ action[:3]
        return np.clip(action, -1., 1.).astype(np.float32)
