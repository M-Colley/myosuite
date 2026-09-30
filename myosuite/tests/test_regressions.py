from types import SimpleNamespace

import mujoco
import numpy as np

from myosuite.envs.env_base import MujocoEnv
from myosuite.envs.heightfields import TrackField, TrackTypes
from myosuite.envs.myo.myobase.reach_v0 import ReachEnvV0
from myosuite.robot.robot import Robot

MODEL_XML = """
<mujoco>
  <worldbody>
    <body name="root">
      <freejoint name="root"/>
      <geom type="sphere" size="0.1"/>
      <body name="link" pos="0 0 0.2">
        <joint name="hinge" type="hinge" range="-1 1"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="hinge" ctrlrange="-2 2"/>
  </actuator>
  <sensor>
    <jointpos name="hinge_pos" joint="hinge"/>
  </sensor>
</mujoco>
"""


def test_batched_reach_rewards_use_each_trajectory_observation():
    env = SimpleNamespace(
        mj_model=SimpleNamespace(na=2),
        far_th=0.02,
        dt=0.1,
        tip_sids=[0],
        rwd_keys_wt={"reach": 1.0, "bonus": 4.0, "penalty": 50.0},
    )
    obs_dict = {
        "reach_err": np.array(
            [
                [[0.01, 0.0, 0.0], [0.2, 0.0, 0.0]],
                [[0.03, 0.0, 0.0], [0.4, 0.0, 0.0]],
            ]
        ),
        "act": np.array(
            [
                [[1.0, 0.0], [0.0, 2.0]],
                [[3.0, 4.0], [6.0, 8.0]],
            ]
        ),
        "time": np.array([[[0.0], [1.0]], [[1.0], [1.0]]]),
    }

    rewards = ReachEnvV0.get_reward_dict(env, obs_dict)

    np.testing.assert_array_equal(
        rewards["done"], [[False, True], [True, True]]
    )
    np.testing.assert_array_equal(
        rewards["solved"], [[True, False], [False, False]]
    )
    np.testing.assert_allclose(
        rewards["act_reg"], [[-0.5, -1.0], [-2.5, -5.0]]
    )


def test_state_restore_preserves_simulation_state_without_stepping():
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    main_data = mujoco.MjData(model)
    robot_data = mujoco.MjData(model)
    main_data.qpos[:] = [0.2, -0.1, 0.9, 1.0, 0.0, 0.0, 0.0, 0.35]
    main_data.qvel[:] = np.arange(model.nv) * 0.01
    main_data.ctrl[:] = 0.75
    main_data.time = 1.25
    mujoco.mj_forward(model, main_data)
    env = SimpleNamespace(
        mj_model=model,
        mj_data=main_data,
        obsd_mj_model=model,
        obsd_mj_data=main_data,
        robot=SimpleNamespace(mj_model=model, mj_data=robot_data),
    )
    state = MujocoEnv.get_env_state(env)

    main_data.qpos[:] = 0
    main_data.qvel[:] = 0
    main_data.ctrl[:] = 0
    robot_data.qpos[:] = -1
    robot_data.qvel[:] = -1
    robot_data.ctrl[:] = -1
    main_data.time = 9.0

    MujocoEnv.set_env_state(env, state)

    for data in (main_data, robot_data):
        np.testing.assert_allclose(data.qpos, state["qpos"])
        np.testing.assert_allclose(data.qvel, state["qvel"])
        np.testing.assert_allclose(data.ctrl, state["ctrl"])
        assert data.time == state["time"]


def test_environment_state_snapshots_are_opt_in():
    env = SimpleNamespace(
        include_env_state=False,
        visual_dict={},
        obs_dict={"time": np.array([1.0])},
        rwd_dict={
            "dense": np.array([0.0]),
            "sparse": np.array([0.0]),
            "solved": np.array([False]),
            "done": np.array([False]),
        },
        proprio_dict={},
        get_env_state=lambda: {"qpos": np.array([1.0])},
    )

    assert "state" not in MujocoEnv.get_env_infos(env)
    assert (
        MujocoEnv.get_env_infos(env, include_state=True)["state"]["qpos"][0]
        == 1.0
    )


def test_reset_seed_propagates_to_stochastic_components():
    class Fatigue:
        def seed(self, seed):
            self.np_random = np.random.default_rng(seed)

    class SeedableEnv(SimpleNamespace):
        def seed(self, seed=None):
            return MujocoEnv.seed(self, seed)

    env = SeedableEnv(
        input_seed=None,
        robot=SimpleNamespace(np_random=None),
        muscle_fatigue=Fatigue(),
        ref=SimpleNamespace(np_random=None),
    )

    def sample(seed):
        kwargs = {"seed": seed}
        MujocoEnv._reseed_for_reset(env, kwargs)
        assert kwargs == {}
        return np.array(
            [
                env.np_random.random(),
                env.robot.np_random.random(),
                env.muscle_fatigue.np_random.random(),
                env.ref.np_random.random(),
            ]
        )

    first = sample(13)
    np.testing.assert_allclose(sample(13), first)
    assert not np.allclose(sample(14), first)
    assert env.robot.np_random is env.np_random
    assert env.ref.np_random is env.np_random


def test_robot_qpos_address_and_velocity_normalization(tmp_path):
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    config_path = tmp_path / "robot_config.txt"
    config_path.write_text(
        repr(
            {
                "device": {
                    "sensor": [{"name": "hinge_pos", "hdr_id": 0}],
                    "actuator": [
                        {
                            "name": "motor",
                            "hdr_id": 0,
                            "pos_range": [-1.0, 1.0],
                            "vel_range": [-2.0, 4.0],
                        }
                    ],
                }
            }
        )
    )
    robot = Robot.__new__(Robot)
    robot.robot_config = None
    robot.is_hardware = False
    config = robot.configure_robot(model, str(config_path))
    actuator = config["device"]["actuator"][0]
    hinge_id = model.joint("hinge").id
    assert actuator["data_id"] == model.jnt_qposadr[hinge_id]
    assert actuator["data_id"] != model.jnt_dofadr[hinge_id]

    robot.robot_config = config
    robot._act_mode = "vel"
    normalized = robot.normalize_actions(np.array([0.5]), unnormalize=True)
    np.testing.assert_allclose(normalized, [2.5])
    recovered = robot.normalize_actions(normalized)
    np.testing.assert_allclose(recovered, [0.5])


def test_trackfield_mixed_generation_covers_final_rows():
    class DummyTrack:
        _fill_terrain = TrackField._fill_terrain

        def __init__(self):
            self.reset_type = "random_mixed"
            self.terrain_type = None
            self.nrow = 600
            self.ncol = 4
            self.rng = np.random.default_rng(5)
            self.hfield = SimpleNamespace(
                data=np.zeros((self.nrow, self.ncol))
            )
            difficulties = np.ones(24)
            self.stairs_difficulties = difficulties
            self.hills_difficulties = difficulties
            self.rough_difficulties = difficulties

            def mark_rows(start, end, index):
                self.hfield.data[start:end] = index + 1

            self._compute_stairs_track = mark_rows
            self._compute_hilly_track = mark_rows
            self._compute_rough_track = mark_rows

    track = DummyTrack()
    TrackField._fill_terrain(track)

    assert track.terrain_type is TrackTypes.MIXED
    assert np.all(track.hfield.data > 0)


def test_rough_track_uses_reproducible_generator():
    class DummyTrack:
        _compute_rough_track = TrackField._compute_rough_track

        def __init__(self, seed):
            self.rng = np.random.default_rng(seed)
            self.rough_difficulties = [0.5]
            self.ncol = 4
            self.hfield = SimpleNamespace(data=np.zeros((8, self.ncol)))

    tracks = [DummyTrack(seed) for seed in (11, 11, 12)]
    for track in tracks:
        TrackField._compute_rough_track(track, 0, 8, 0)

    np.testing.assert_allclose(tracks[0].hfield.data, tracks[1].hfield.data)
    assert not np.allclose(tracks[0].hfield.data, tracks[2].hfield.data)
