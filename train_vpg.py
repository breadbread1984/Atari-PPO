#!/usr/bin/python3

import os;
import numpy as np;
import cv2;
import tensorflow as tf;
from atari_py import ALEInterface;
from atari_py import get_game_path;

class PolicyModel(tf.keras.Model):

    def __init__(self, legal_actions = None):

        assert type(legal_actions) is np.ndarray and len(legal_actions) > 0;
        super(PolicyModel,self).__init__();
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size = [3,3], padding = 'same');
        self.relu1 = tf.keras.layers.ReLU();
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size = [2,2], strides = [2,2], padding = 'valid');
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size = [3,3], padding = 'same');
        self.relu2 = tf.keras.layers.ReLU();
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size = [2,2], strides = [2,2], padding = 'valid');
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size = [3,3], padding = 'same');
        self.relu3 = tf.keras.layers.ReLU();
        self.reduce = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis = [1,2]));
        self.dense4 = tf.keras.layers.Dense(200);
        self.dropout4 = tf.keras.layers.Dropout(0.5);
        self.relu4 = tf.keras.layers.ReLU();
        # output P(a|s)
        self.dense5 = tf.keras.layers.Dense(legal_actions.shape[0], activation = tf.math.sigmoid);
        self.exp = tf.keras.layers.Lambda(lambda x: tf.math.exp(x));
        # output V(s)
        self.dense6 = tf.keras.layers.Dense(1, activation = tf.math.sigmoid);
        
    def call(self, input):
        
        result = self.conv1(input);
        result = self.relu1(result);
        result = self.maxpool1(result);
        result = self.conv2(result);
        result = self.relu2(result);
        result = self.maxpool2(result);
        result = self.conv3(result);
        result = self.relu3(result);
        result = self.reduce(result);
        result = self.dense4(result);
        result = self.dropout4(result);
        result = self.relu4(result);
        logP = self.dense5(result);
        V = self.dense6(result);
        return V, logP;

class VPG(object):
    
    def __init__(self):
        
        self.ale = ALEInterface();
        self.ale.loadROM(get_game_path('boxing'));
        self.legal_actions = self.ale.getMinimalActionSet();
        self.policyModel = PolicyModel(self.legal_actions);
        #load model
        if True == os.path.exists('model'): self.policyModel.load_weights('./model/vpg_model');
        self.status_size_ = 4
        self.gamma_ = 0.8;
        
    def status2tensor(self,status):
        
        status = tf.convert_to_tensor(status, dtype = tf.float32);
        status = tf.transpose(status,[1,2,0]);
        status = tf.expand_dims(status,0);
        return status;
        
    def PlayOneEpisode(self):
        
        self.ale.reset_game();
        trajectory = list();
        status = list();
        # initial status
        for i in range(self.status_size_):
            current_frame = cv2.resize(self.ale.getScreenGrayscale(),(84,84));
            status.append(current_frame);
            assert False == self.ale.game_over();
        # play until game over
        while False == self.ale.game_over():
            # display screen
            cv2.imshow('screen',self.ale.getScreenRGB());
            cv2.waitKey(10);
            # choose action 
            input = self.status2tensor(status);
            V,logP = self.policyModel(input);
            action_index = tf.random.categorical(tf.math.exp(logP),1);
            reward = self.ale.act(self.legal_actions[action_index]);
            status.append(cv2.resize(self.ale.getScreenGrayscale(),(84,84)));
            trajectory.append((status[0:self.status_size_],action_index,reward,status[1:]));
            status = status[1:];
        total_reward = 0;
        for status in reversed(trajectory):
            total_reward = status[2] + self.gamma_ * total_reward;
        return trajectory, total_reward;
    
    def train(self, loop_time = 1000):
        
        optimizer = tf.keras.optimizers.Adam(1e-4);
        # setup checkpoint and log utils
        checkpoint = tf.train.Checkpoint(model = self.policyModel, optimizer = optimizer, optimizer_step = optimizer.iterations);
        checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
        log = tf.summary.create_file_writer('checkpoints');
        for i in range(loop_time):
            trajectory, total_reward = self.PlayOneEpisode();
            avg_policy_loss = tf.keras.metrics.Mean(name = 'policy loss', dtype = tf.float32);
            avg_value_loss = tf.keras.metrics.Mean(name = 'value loss', dtype = tf.float32);
            for status in trajectory:
                # policy loss
                with tf.GradientTape() as tape:
                    Vt, logPt = self.policyModel(self.status2tensor(status[0]));
                    Vtp1, logPtp1 = self.policyModel(self.status2tensor(status[3]));
                    action_mask = tf.one_hot(status[1],len(self.legal_actions));
                    log_probs = tf.math.reduce_sum(action_mask * logPt, axis = 1);
                    advantage = -Vt + status[2] + self.gamma_ * Vtp1;
                    policy_loss = -tf.math.reduce_mean(log_probs * advantage);
                    value_loss = tf.math.squared_difference(Vt, total_reward);
                    loss = policy_loss + value_loss;
                    avg_policy_loss.update_state(policy_loss);
                    avg_value_loss.update_state(value_loss);
                # write loss to summary
                if tf.equal(optimizer.iterations % 100, 0):
                    with log.as_default():
                        tf.summary.scalar('policy loss',avg_policy_loss.result(), step = optimizer.iterations);
                        tf.summary.scalar('value loss',avg_value_loss.result(), step = optimizer.iterations);
                    avg_policy_loss.reset_states();
                    avg_value_loss.reset_states();
                # train policy and value
                grads = tape.gradient(loss,self.policyModel.variables);
                optimizer.apply_gradients(zip(grads,self.policyModel.variables));
            # save model every episode
            checkpoint.save(os.path.join('checkpoints','ckpt'));
        # save final model
        if False == os.path.exists('model'): os.mkdir('model');
        #tf.saved_model.save(self.policyModel,'./model/vpg_model');
        self.policyModel.save_weights('./model/vpg_model');

def main():

    vpg = VPG();
    vpg.train(100);

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
