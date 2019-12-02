from gym.envs.registration import register

register(id='BackCheetah-v0', entry_point='gym_backcheetah.envs:BackCheetahEnv',max_episode_steps=1000, reward_threshold=4800.0,)
