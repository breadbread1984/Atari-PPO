#!/usr/bin/python3

import tensorflow as tf;

class GAE:
    # this class is used for getting advantages over all time before the end
    # of an episode

    def __init__(self):

        self.memory = list();

    def push(self, reward, value):

        self.memory.append((reward,value));

    def advantages(self, time = 0, gam = 0.5):

        # get advantages from the given time to the end of episode
        memory_from = self.memory[slice(time,len(self.memory))];
        rewards_from, values_from = zip(*memory_from);
        rewards_from.append(0);
        values_from.append(0);
        # advantage_t = r_t + gamma * V(s_t+1) - V(s_t)
        # t ranges from the given time to the last time when the
        # status is not a terminal status
        advantages = rewards[:-1] + gam * values[1:] - values[:-1];
        return advantages;

    def getGAE(self. gam = 0.5, lam = 0.5):

        rewards, values = zip(*self.memory);
        gaes = list();
        for t in range(len(self.memory)):
            advs = tf.convert_to_tensor(self.advantages(t,gam), dtype = tf.float32);
            weights = tf.convert_to_tensor([(gam * lam)**i for i in range(len(advs))], dtype = tf.float32);
            gae = tf.math.reduce_sum(advs * weights);
            gaes.append(gae);
        assert len(gaes) == len(self.memory);
        return gaes;

