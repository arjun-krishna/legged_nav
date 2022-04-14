# Isaac Gym environments for Legged Robot Navigation

building on top of [legged_gym](https://github.com/leggedrobotics/legged_gym) environments for 
point goal navigation are created.

# Installation
1. create rlgpu env from [Isaac Gym](https://developer.nvidia.com/isaac-gym) Preview 3
2. ```bash
    pip install -r requirements.txt
   ```

# Fun commands
```bash
python run.py task=A1FlatLocoLidar num_envs=1 test=True task.commands.use_key_events=True task.env.debug_viz=True task.env.episode_length_s=120
```