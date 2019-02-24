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
        self.bn1 = tf.keras.layers.BatchNormalization();
        self.relu1 = tf.keras.layers.ReLU();
        self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size = [2,2], strides = [2,2], padding = 'valid');
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size = [3,3], padding = 'same');
        self.bn2 = tf.keras.layers.BatchNormalization();
        self.relu2 = tf.keras.layers.ReLU();
        self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size = [2,2], strides = [2,2], padding = 'valid');
        self.conv3 = tf.keras.layers.Conv2D(256, kernel_size = [3,3], padding = 'same');
        self.bn3 = tf.keras.layers.BatchNormalization();
        self.relu3 = tf.keras.layers.ReLU();
        self.reduce = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis = [1,2]));
        self.dense4 = tf.keras.layers.Dense(200);
        self.dropout4 = tf.keras.layers.Dropout(0.5);
        self.bn4 = tf.keras.layers.BatchNormalization();
        self.relu4 = tf.keras.layers.ReLU();
        # output P(a|s)
        self.dense5 = tf.keras.layers.Dense(legal_actions.shape[0], activation = tf.math.sigmoid);
        self.exp = tf.keras.layers.Lambda(lambda x: tf.math.exp(x));
        # output V(s)
        self.dense6 = tf.keras.layers.Dense(1, activation = tf.math.sigmoid);
        
    def call(self, input):
        
        result = self.conv1(input);
        result = self.bn1(result);
        result = self.relu1(result);
        result = self.maxpool1(result);
        result = self.conv2(result);
        result = self.bn2(result);
        result = self.relu2(result);
        result = self.maxpool2(result);
        result = self.conv3(result);
        result = self.bn3(result);
        result = self.relu3(result);
        result = self.reduce(result);
        result = self.dense4(result);
        result = self.dropout4(result);
        result = self.bn4(result);
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
        return trajectory;
    
    def train(self, loop_time = 1000):
        
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4);
        # setup checkpoint and log utils
        checkpoint = tf.train.Checkpoint(model = self.policyModel, optimizer = optimizer, optimizer_step = tf.compat.v1.train.get_or_create_global_step());
        checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
        log = tf.summary.create_file_writer('checkpoints');
        log.set_as_default();
        for i in range(loop_time):
            trajectory = self.PlayOneEpisode();
            total_reward = 0;
            for status in reversed(trajectory):
                # update total reward
                total_reward = status[2] + self.gamma_ * total_reward;
                # policy loss
                with tf.GradientTape() as tape:
                    Vt, logPt = self.policyModel(self.status2tensor(status[0]));
                    Vtp1, logPtp1 = self.policyModel(self.status2tensor(status[4]));
                    action_mask = tf.one_hot(status[1],len(self.legal_actions));
                    log_probs = tf.math.reduce_sum(action_mask * logPt, axis = 1);
                    advantage = -Vt + status[2] + self.gamma_ * Vtp1;
                    loss = -tf.reduce_mean(log_probs * advantage);
                # write policy loss
                with tf.summary.record_summaries_every_n_global_steps(1,global_step = tf.train.get_global_step()):
                    tf.contrib.summary.scalar('policy loss',loss);
                # train policy
                grads = tape.gradient(loss,self.policyModel.variables);
                optimizer.apply_gradients(zip(grads,model.variables), global_step = tf.train.get_global_step());
            for status in trajectory:
                # value loss
                with tf.GradientTape() as tape:
                    Vt, logPt = self.policyModel(self.status2tensor(status[0]));
                    loss = tf.math.squared_difference(Vt, total_reward);
                # write value loss
                with tf.summary.record_summaries_every_n_global_steps(1,global_step = tf.train.get_global_step()):
                    tf.contrib.summary.scalar('value loss',loss);
                # train value
                grads = tape.gradient(loss,self.policyModel.variables);
                optimizer.apply_gradients(zip(grads,model.variables), global_step = tf.train.get_global_step());
            # save model every episode
            checkpoint.save(os.path.join('checkpoints','ckpt'));
        # save final model
        if False == os.path.exists('model'): os.mkdir('model');
        self.policyModel.save_weights('./model/vpg_model');

def main():

    vpg = VPG();
    vpg.train();

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();

