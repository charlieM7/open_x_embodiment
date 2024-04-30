# Used to test robosuite
# Base code from https://robosuite.ai/docs/quickstart.html

import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from PIL import Image
import os

def rotate_180(image):
    image = np.flipud(image)
    # return np.fliplr(image)
    return image
    # return np.rot90(image, 2)

# Check if MUJOCO_EGL_DEVICE_ID is set
if "MUJOCO_EGL_DEVICE_ID" in os.environ:
    mujoco_egl_device_id = int(os.environ["MUJOCO_EGL_DEVICE_ID"])
    print("MUJOCO_EGL_DEVICE_ID is set to:", mujoco_egl_device_id)
else:
    print("MUJOCO_EGL_DEVICE_ID is not set")

# Check if CUDA_VISIBLE_DEVICES is set
if "CUDA_VISIBLE_DEVICES" in os.environ:
    # os.environ.pop("CUDA_VISIBLE_DEVICES")
    cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    print("CUDA_VISIBLE_DEVICES is set to:", cuda_visible_devices)
else:
    print("CUDA_VISIBLE_DEVICES is not set")

if "MUJOCO_GPU_RENDERING" in os.environ:
    mujoco_gpu_rendering = os.environ["MUJOCO_GPU_RENDERING"]
    print("MUJOCO_GPU_RENDERING is set to:", mujoco_gpu_rendering)
else:
    print("MUJOCO_GPU_RENDERING is not set")

config = load_controller_config(default_controller='OSC_POSE')
# create environment instance
env = suite.make(
    env_name="Door", 
    robots="Panda", 
    controller_configs=config,
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview" 
)

# reset the environment
env.reset()
env.viewer.set_camera(camera_id=1)
low, high = env.action_spec
start_obs = env.observation_spec()

action = np.random.uniform(low, high)
action = np.array([0,1,0,0,0,0,-1])

# import pdb; pdb.set_trace()
while(True):
    frames = []
    for i in range(10):
        # action = np.random.uniform(low, high)
        obs, reward, done, info = env.step(action)  # take action in the environment

        rotated = rotate_180(obs['agentview_image'])
        image = Image.fromarray(rotated)
        image.save(f'frame_{i}.png')
        import pdb; pdb.set_trace()
        env.render()  # render on display
    