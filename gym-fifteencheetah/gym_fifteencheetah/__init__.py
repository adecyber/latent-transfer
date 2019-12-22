from gym.envs.registration import register

register(id='FifteenCheetah-v0', entry_point='gym_fifteencheetah.envs:FifteenCheetahEnv',max_episode_steps=1000, reward_threshold=4800.0,)
