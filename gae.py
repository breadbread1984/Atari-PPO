#!/usr/bin/python3

import tensorflow as tf;

class GAE:
    # this class is used for getting advantages over all time before the end
    # of an episode

    def __init__(self):

        self.memory = list();

    def push(self, reward, value):

        # add r_t and V(s_t)
        self.memory.append((reward,value));

    def advantages(self, time = 0, gam = 0.5):

        # get advantages from the given time to the end time of episode
        memory_from = self.memory[slice(time,len(self.memory))];
        rewards_from, values_from = zip(*memory_from);
        rewards_from = tf.convert_to_tensor(rewards_from + (0,), dtype = tf.float32);
        values_from = tf.convert_to_tensor(values_from + (0,), dtype = tf.float32);
        # advantage_t = r_t + gamma * V(s_t+1) - V(s_t)
        # t ranges from the given time to the last time when the
        # status is not a terminal status
        advantages = rewards_from[:-1] + gam * values_from[1:] - values_from[:-1];
        return advantages;

    def gaes(self, gam = 0.5, lam = 0.5):

        # get general advantages estimate over all times
        rewards, values = zip(*self.memory);
        gaes = list();
        for t in range(len(self.memory)):
            advs = self.advantages(t,gam);
            weights = tf.convert_to_tensor([(gam * lam)**i for i in range(len(advs))], dtype = tf.float32);
            gae = tf.math.reduce_sum(advs * weights);
            gaes.append(gae);
        assert len(gaes) == len(self.memory);
        return gaes;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    gae = GAE();
    gae.push(1,2);
    gae.push(2,1);
    gae.push(1,3);
    print(gae.gaes());
