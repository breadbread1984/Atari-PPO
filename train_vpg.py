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
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size = [8,8], strides = [4,4], padding = 'valid');
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size = [4,4], strides = [2,2], padding = 'valid');
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size = [3,3], padding = 'valid');
        self.reduce = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis = [1,2]));
        self.dense4 = tf.keras.layers.Dense(512);
        self.dropout4 = tf.keras.layers.Dropout(0.5);
        self.dense5 = tf.keras.layers.Dense(512);
        self.dropout5 = tf.keras.layers.Dropout(0.5);
        self.relu = tf.keras.layers.ReLU();
        # output P(a|s)
        self.dense6 = tf.keras.layers.Dense(legal_actions.shape[0], activation = tf.math.sigmoid);
        self.exp = tf.keras.layers.Lambda(lambda x: tf.math.exp(x));
        # output V(s)
        self.dense7 = tf.keras.layers.Dense(1, activation = tf.math.sigmoid);
        
    def call(self, input):
        
        result = self.conv1(input);
        result = self.relu(result);
        result = self.conv2(result);
        result = self.relu(result);
        result = self.conv3(result);
        result = self.relu(result);
        result = self.reduce(result);
        result = self.dense4(result);
        result = self.dropout4(result);
        result = self.relu(result);
        result = self.dense5(result);
        result = self.dropout5(result);
        result = self.relu(result);
        logP = self.dense6(result);
        P = self.exp(logP);
        V = self.dense7(result);
        return V, P;

class VPG(object):
    
    def __init__(self):
        
        self.ale = ALEInterface();
        self.ale.loadROM(get_game_path('boxing'));
        self.legal_actions = self.ale.getMinimalActionSet();
        self.policyModel = PolicyModel(self.legal_actions);
        #load model
        if True == os.path.exists('model'): self.policyModel.load_weights('./model/vpg_model');
        self.status_size_ = 4
        self.gamma_ = 1; #the reward it too small
        
    def status2tensor(self,status):
        
        status = tf.convert_to_tensor(status, dtype = tf.float32);
        status = tf.transpose(status,[1,2,0]);
        status = tf.expand_dims(status,0);
        return status;
        
    def preprocess(self, image):
        
        frame = image[25:185,:,:];
        frame = cv2.resize(frame,(84,84)) / 255.0;
        return frame;
        
    def PlayOneEpisode(self):
        
        self.ale.reset_game();
        trajectory = list();
        status = list();
        # initial status
        for i in range(self.status_size_):
            current_frame = self.preprocess(self.ale.getScreenGrayscale());
            status.append(current_frame);
            assert False == self.ale.game_over();
        # play until game over
        while False == self.ale.game_over():
            # display screen
            cv2.imshow('screen',self.ale.getScreenRGB());
            cv2.waitKey(10);
            # choose action 
            input = self.status2tensor(status);
            V, P = self.policyModel(input);
            action_index = tf.random.categorical(P,1);
            reward = 0;
            for i in range(self.status_size_):
                reward += self.ale.act(self.legal_actions[action_index]);
            current_frame = self.preprocess(self.ale.getScreenGrayscale());
            status.append(current_frame);
            game_over = self.ale.game_over();
            trajectory.append((status[0:self.status_size_],action_index,reward,status[1:],game_over));
            status = status[1:];
        total_reward = 0;
        for status in reversed(trajectory):
            total_reward = status[2] + self.gamma_ * total_reward;
        return trajectory, total_reward;
    
    def train(self, loop_time = 1000):
        
        optimizer = tf.keras.optimizers.Adam(1e-3);
        # setup checkpoint and log utils
        checkpoint = tf.train.Checkpoint(model = self.policyModel, optimizer = optimizer, optimizer_step = optimizer.iterations);
        checkpoint.restore(tf.train.latest_checkpoint('checkpoints_vpg'));
        log = tf.summary.create_file_writer('checkpoints_vpg');
        for i in range(loop_time):
            trajectory, total_reward = self.PlayOneEpisode();
            avg_policy_loss = tf.keras.metrics.Mean(name = 'policy loss', dtype = tf.float32);
            avg_value_loss = tf.keras.metrics.Mean(name = 'value loss', dtype = tf.float32);
            for status in trajectory:
                # policy loss
                with tf.GradientTape() as tape:
                    Vt, Pt = self.policyModel(self.status2tensor(status[0]));
                    Vtp1, Ptp1 = self.policyModel(self.status2tensor(status[3]));
                    action_mask = tf.one_hot(status[1],len(self.legal_actions));
                    log_probs = tf.math.reduce_sum(action_mask * tf.math.log(Pt), axis = 1);
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
            checkpoint.save(os.path.join('checkpoints_vpg','ckpt'));
        # save final model
        if False == os.path.exists('model'): os.mkdir('model');
        #tf.saved_model.save(self.policyModel,'./model/vpg_model');
        self.policyModel.save_weights('./model/vpg_model');

def main():

    vpg = VPG();
    vpg.train(1000);
    #vpg.PlayOneEpisode();

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
