from gym.envs.registration import register

register(id='TwentyCheetah-v0', entry_point='gym_twentycheetah.envs:TwentyCheetahEnv',max_episode_steps=1000, reward_threshold=4800.0,)
