from gym.envs.registration import register

register(id='CheetahVel-v0', entry_point='gym_cheetahvel.envs:CheetahVelEnv',max_episode_steps=1000, reward_threshold=4800.0,)
