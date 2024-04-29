"""Runs inference with a RT-1 model."""

import copy

from absl import app
from absl import flags

from flax.training import checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

import rt1
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from PIL import Image
import os
import matplotlib.pyplot as plt
import imageio

jax.config.update('jax_default_device', jax.devices('cpu')[0])

_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path', "/usr/local/data/c.mori7/Repos/open_x_embodiment/rt_1_x_jax/checkpoint", 'Path to checkpoint.'
)
flags.mark_flag_as_required('checkpoint_path')


class RT1Policy:
    """Runs inference with a RT-1 policy."""
    def __init__(
        self,
        checkpoint_path=None,
        model=rt1.RT1(),
        variables=None,
        seqlen=15,
        rng=None,
    ):
        """Initializes the policy.
        Args:
        checkpoint_path: A checkpoint point from which to load variables. Either
            this or variables must be provided.
        model: A nn.Module to use for the policy. Must match with the variables
            provided by checkpoint_path or variables.
        variables: If provided, will use variables instead of loading from
            checkpoint_path.
        seqlen: The history length to use for observations.
        rng: a jax.random.PRNGKey to use for the random number generator.
        """
        if not variables and not checkpoint_path:
            raise ValueError(
                'At least one of `variables` or `checkpoint_path` must be defined.'
            )
        
        self.model = model
        self._checkpoint_path = checkpoint_path
        self.seqlen = seqlen

        self._run_action_inference_jit = jax.jit(self._run_action_inference)

        if rng is None:
            self.rng = jax.random.PRNGKey(0)
        else:
            self.rng = rng

        if variables:
            self.variables = variables
        else:
            state_dict = checkpoints.restore_checkpoint(checkpoint_path, None)
            variables = {
                'params': state_dict['params'],
                'batch_stats': state_dict['batch_stats'],
            }
            self.variables = variables

    def _run_action_inference(self, observation, rng):
        """A jittable function for running inference."""

        # We add zero action tokens so that the shape is (seqlen, 11).
        # Note that in the vanilla RT-1 setup, where 
        # `include_prev_timesteps_actions=False`, the network will not use the
        # input tokens and instead uses zero action tokens, thereby not using the
        # action history. We still pass it in for simplicity.
        act_tokens = jnp.zeros((1, 6, 11))

        # Add a batch dim to the observation.
        batch_obs = jax.tree_map(lambda x: jnp.expand_dims(x, 0), observation)

        _, random_rng = jax.random.split(rng)

        output_logits = self.model.apply(
            self.variables,
            batch_obs,
            act=None,
            act_tokens=act_tokens,
            train=False,
            rngs={'random': random_rng},
        )

        time_step_tokens = (
            self.model.num_image_tokens + self.model.num_action_tokens
        )
        output_logits = jnp.reshape(
            output_logits, (1, self.seqlen, time_step_tokens, -1)
        )
        action_logits = output_logits[:, -1, ...]
        action_logits = action_logits[:, self.model.num_image_tokens - 1 : -1]

        action_logp = jax.nn.softmax(action_logits)
        action_token = jnp.argmax(action_logp, axis=-1)

        # Detokenize the full action sequence.
        detokenized = rt1.detokenize_action(
            action_token, self.model.vocab_size, self.model.world_vector_range
        )

        detokenized = jax.tree_map(lambda x: x[0], detokenized)

        return detokenized

    def action(self, observation):
        """Outputs the action given observation from the env."""
        # Assume obs has no batch dimensions.
        observation = copy.deepcopy(observation)

        # Jax does not support string types, so remove it from the dict if it
        # exists.
        if 'natural_language_instruction' in observation:
            del observation['natural_language_instruction']

        image = observation['image']
        # Resize using TF image resize to avoid any issues with using different
        # resize implementation, since we also use tf.image.resize in the data
        # pipeline. Also scale image to [0, 1].
        image = tf.image.resize(image, (300, 300)).numpy()
        image /= 255.0
        observation['image'] = image

        self.rng, rng = jax.random.split(self.rng)
        action = self._run_action_inference_jit(
            observation, rng
        )
        action = jax.device_get(action)

        # Use the base pose mode if the episode if the network outputs an invalid
        # `terminate_episode` action.
        if np.sum(action['terminate_episode']) == 0:
            action['terminate_episode'] = np.zeros_like(action['terminate_episode'])
            action['terminate_episode'][-1] = 1
        
        return action


class RobosuiteEnv:
    def __init__(
        self,
        env_name="Lift",
        robots="Panda",
        controller_config=None,
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        reward_shaping=True, 
        horizon=300,  
    ):
        #robosuite env variables
        self.env_name = env_name
        self.robots = robots
        self.controller_config = controller_config
        self.has_renderer = has_renderer
        self.has_offscreen_renderer = has_offscreen_renderer
        self.use_camera_obs = use_camera_obs
        self.camera_names = camera_names
        self.reward_shaping = reward_shaping
        self.horizon = horizon

        #envs for take action
        self.frames = []
        self.episode_rewards = []
        self.done = False

        #video recording envs
        self.record_video = False
        self.video_filename = None
        self.video_writer = None

        if self.controller_config is None:
            self.controller_config = load_controller_config(default_controller='OSC_POSE')

        self.env = suite.make(
            env_name= self.env_name,
            robots= self.robots,
            controller_configs=self.controller_config,
            has_renderer=self.has_renderer,
            has_offscreen_renderer=self.has_offscreen_renderer,
            use_camera_obs=self.use_camera_obs,
            camera_names = self.camera_names,
            reward_shaping = self.reward_shaping,
            horizon = self.horizon
        )

        # reset the environment
        self.env.reset()
        self.env.viewer.set_camera(camera_id=0)

    # code to record episode
    def start_vid(self, filename):
        self.record_video = True
        self.video_filename = filename
        self.video_writer = imageio.get_writer(filename, fps=30)

    def stop_vid(self):
        self.record_video = False
        if self.video_writer:
            self.video_writer.close()

    # image preprocessing
    def flip(self, image):
        image = np.flipud(image)
        return image

    def take_action(self, action):
        if action is None:
            obs = self.env.observation_spec()
            for i in range(15):
                rotated = self.flip(obs[self.camera_names + '_image'])
                self.frames.append(rotated)
                # self.env.render() 
            self.frames = jnp.array(self.frames)

        else:
            obs, reward, done, info = self.env.step(action)  # take action in the environment
            self.episode_rewards.append(reward)
            self.done = done
            rotated = self.flip(obs[self.camera_names + '_image'])
            self.frames = self.frames.at[:-1].set(self.frames[1:])
            self.frames = self.frames.at[-1].set(rotated)
            # self.env.render() 

        if self.record_video:
            self.video_writer.append_data(rotated)
        
        return self.frames.astype(np.float32)

    def plot_reward(self):
        filename = f'figures/reward_{self.horizon}.txt'
        with open(filename, 'w') as file:
            for reward in self.episode_rewards:
                file.write(str(reward) + '\n')
        # Plot episode rewards
        # values, base = np.histogram(self.episode_rewards, bins=40)
        cumulative_rewards = np.cumsum(self.episode_rewards)
        plt.plot(cumulative_rewards)
        # plt.plot(self.episode_rewards)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.title('Episode Cumulative Rewards')
        plt.grid(True)

        # Save plot as an image file
        plt.savefig(f'figures/episode_{self.horizon}_rewards.png')
        plt.show()


def main(argv):
    del argv
    sequence_length = 15
    num_action_tokens = 11
    layer_size = 256
    vocab_size = 512
    num_image_tokens = 81

    rt1x_model = rt1.RT1(
        num_image_tokens=num_image_tokens,
        num_action_tokens=num_action_tokens,
        layer_size=layer_size,
        vocab_size=vocab_size,
        # Use token learner to reduce tokens per image to 81.
        use_token_learner=True,
        # RT-1-X uses (-2.0, 2.0) instead of (-1.0, 1.0).
        world_vector_range=(-2.0, 2.0),
    )

    policy = RT1Policy(
        checkpoint_path=_CHECKPOINT_PATH.value,
        model=rt1x_model,
        seqlen=sequence_length,
    )

    terminate = 0
    robot_action = None
    env.start_vid("figures/episode.mp4")

    while(not terminate and not env.done):
        frames = env.take_action(robot_action)
        loaded_embeddings = np.load('embeddings/prompt_embeddings7.npy')

        obs = {
            'image': frames,
            'natural_language_embedding': np.tile(loaded_embeddings, (15, 1)),
        }

        actions = policy.action(obs)
        # uncomment to flip y-axis 
        # x = np.array([actions['world_vector'][0]*1, actions['world_vector'][1]*-1, actions['world_vector'][2]])
        # y = np.array([actions['rotation_delta'][0]*1, actions['rotation_delta'][1]*-1, actions['rotation_delta'][2]])
        # robot_action = np.concatenate((x, y, actions['gripper_closedness_action']))

        robot_action = np.concatenate((actions['world_vector'], actions['rotation_delta'], actions['gripper_closedness_action']))
        print(robot_action)

        terminate = 1 if all(actions["terminate_episode"] == [1,0,0]) else 0
        print(str(actions["terminate_episode"]) + " " + str(terminate))
    
    env.plot_reward()
    env.stop_vid()


if __name__ == '__main__':
    app.run(main)
