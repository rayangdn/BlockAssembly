import gym
import numpy as np
from gym import spaces
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.wrappers.recsim_wrapper import DiscreteActionMaskEnv
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.policy.policy import Policy
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

from assembly_env import AssemblyEnv, Action
from tasks import Bridge, Tower, DoubleBridge


torch, nn = try_import_torch()

def encode_action(target_block, target_face, shape, face, offset_x):
    return (((target_block * 4 + target_face) * 2 + shape) * 4 + face) * 11 + offset_x

def decode_action(action_idx):
    offset_x = action_idx % 11
    action_idx //= 11
    face = action_idx % 4
    action_idx //= 4
    shape = action_idx % 2
    action_idx //= 2
    target_face = action_idx % 4
    target_block = action_idx // 4
    return target_block, target_face, shape, face, offset_x


class BlockPlacementEnv(gym.Env):
    def __init__(self, task, config: EnvContext):
        super().__init__()
        self.n_blocks = config.get("n_blocks")

        # NOTE Action(target_block, target_face, shape, face, offset_x)
        self.action_space_size =  (self.n_blocks + 1) * 4 * 2 * 4 * 11

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=0, high=255, shape=config.get("obs_shape"), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(self.action_space_size,), dtype=np.float32),
        })

        self.action_space = spaces.Discrete(self.action_space_size)

        self.physics_env = AssemblyEnv(
            task=task, 
            max_blocks=self.n_blocks,
            xlim=config.get("xlim"),
            zlim=config.get("zlim"),
            img_size=config.get("obs_shape"),
            mu=config.get("mu"),
            density=config.get("density"),
            # state_representation="basic", # TODO after merging environments
            # reward_representation="basic",
        )

        self.reset()

    def reset(self, seed=42):
        self.seed(seed)
        self.physics_env.reset()

        obs = self.physics_env.state_feature.numpy()
        action_mask = self._compute_action_mask()
        return {"obs": obs, "action_mask": action_mask}

    def step(self, action_idx):
        target_block, target_face, shape, face, offset_x = decode_action(action_idx)
        action = Action(target_block=target_block, target_face=target_face, shape=shape, face=face, offset_x=offset_x)

        obs, reward, done = self.physics_env.step(action)

        action_mask = self._compute_action_mask()
        return {"obs": obs, "action_mask": action_mask}, reward, done # , {} # TODO return "info" ?

    def _compute_action_mask(self):
        available_actions = self.physics_env.available_actions(num_block_offsets=self.num_offsets)
        mask = np.full(self.action_space_size, False, dtype=np.float32)
        
        for action in available_actions:
            action_idx = encode_action(action)
            mask[action_idx] = 1.0

        return mask


class MaskedModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.internal_model = FullyConnectedNetwork(
            obs_space.original_space["obs"], action_space, num_outputs, model_config, name + "_internal"
        )

    def forward(self, input_dict, state, seq_lens):
        logits, _ = self.internal_model(input_dict["obs"], state, seq_lens)

        action_mask = input_dict["obs"]["action_mask"]
        inf_mask = torch.clamp(torch.log(action_mask), min=torch.finfo(torch.float32).min)

        return logits + inf_mask, state

    def value_function(self):
        return self.internal_model.value_function()


ModelCatalog.register_custom_model("masked_model", MaskedModel)


def create_dqn_trainer(env_name):
    config = {
        "env": env_name,
        "env_config": {
            "n_blocks": 5,
            "obs_shape": (64, 64),
            "xlim": (-3, 3),
            "zlim": (0, 6),
            "mu": 0.8,
            "density": 1.0
        },
        "model": {
            "custom_model": "masked_model",
        },
        "framework": "torch",
        "num_workers": 0,
        "explore": True,
        "exploration_config": {
            "type": "EpsilonGreedy",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.05,
            "epsilon_timesteps": 10000,
        },
    }
    return DQNTrainer(config=config)


gym.register(
    id="BlockPlacement-v0",
    entry_point=BlockPlacementEnv,
)


trainer = create_dqn_trainer("BlockPlacement-v0")
for i in range(100):
    result = trainer.train()
    print(result["episode_reward_mean"])
trainer.save("./checkpoint")
