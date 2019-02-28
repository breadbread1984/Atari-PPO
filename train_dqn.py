#!/usr/bin/python3

import os;
import numpy as np;
import cv2;
import tensorflow as tf;
from atari_py import ALEInterface;
from atari_py import get_game_path;

class QNet(tf.keras.Model):

    def __init__(self, legal_actions = None):

        assert type(legal_actions) is np.ndarray and len(legal_actions) > 0;
        super(QNet,self).__init__();
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size = [8,8], strides = [4,4], padding = 'valid');
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size = [4,4], strides = [2,2], padding = 'valid');
        self.conv3 = tf.keras.layers.Conv2D(64, kernel_size = [3,3], padding = 'valid');
        self.reduce = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x, axis = [1,2]));
        self.dense4 = tf.keras.layers.Dense(512);
        self.dropout4 = tf.keras.layers.Dropout(0.5);
        self.dense5 = tf.keras.layers.Dense(512);
        self.dropout5 = tf.keras.layers.Dropout(0.5);
        self.relu = tf.keras.layers.ReLU();
        # output Q(s,a)
        self.dense6 = tf.keras.layers.Dense(legal_actions.shape[0], activation = tf.math.sigmoid);
        
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
        Q = self.dense6(result);
        return Q;

class DQN(object):
    
    def __init__(self):
        
        self.ale = ALEInterface();
        self.ale.loadROM(get_game_path('boxing'));
        self.legal_actions = self.ale.getMinimalActionSet();
        self.qnet = QNet(self.legal_actions);
        #load model
        if True == os.path.exists('model'): self.qnet.load_weights('./model/dqn_model');
        self.status_size_ = 4
        self.gamma_ = 1;

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
            Qt = self.qnet(input);
            action_index = tf.random.categorical(tf.math.exp(Qt),1);
            reward = 0;
            for i in range(self.status_size_):
                reward += self.ale.act(self.legal_actions[action_index]);
            current_frame = self.preprocess(self.ale.getScreenGrayscale());
            status.append(current_frame);
            game_over = self.ale.game_over();
            trajectory.append((status[0:self.status_size_],action_index,reward,status[1:],game_over));
            status = status[1:];
        return trajectory;
    
    def train(self, loop_time = 1000):
        
        optimizer = tf.keras.optimizers.Adam(1e-3);
        # setup checkpoint and log utils
        checkpoint = tf.train.Checkpoint(model = self.qnet, optimizer = optimizer, optimizer_step = optimizer.iterations);
        checkpoint.restore(tf.train.latest_checkpoint('checkpoints_dqn'));
        log = tf.summary.create_file_writer('checkpoints_dqn');
        for i in range(loop_time):
            trajectory = self.PlayOneEpisode();
            avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
            for status in trajectory:
                # policy loss
                with tf.GradientTape() as tape:
                    Qt = self.qnet(self.status2tensor(status[0]));
                    Qtp1 = self.qnet(self.status2tensor(status[3]));
                    action_mask = tf.one_hot(status[1],len(self.legal_actions));
                    qt = tf.math.reduce_sum(action_mask * Qt, axis = 1);
                    qtp1 = tf.math.reduce_max(Qtp1, axis = 1);
                    value = status[2] + (self.gamma_ * qtp1 if False == status[4] else 0);
                    loss = tf.math.squared_difference(qt, value);
                    avg_loss.update_state(loss);
                # write loss to summary
                if tf.equal(optimizer.iterations % 100, 0):
                    with log.as_default():
                        tf.summary.scalar('loss',avg_loss.result(), step = optimizer.iterations);
                    avg_loss.reset_states();
                # train policy and value
                grads = tape.gradient(loss,self.qnet.variables);
                optimizer.apply_gradients(zip(grads,self.qnet.variables));
            # save model every episode
            checkpoint.save(os.path.join('checkpoints_dqn','ckpt'));
        # save final model
        if False == os.path.exists('model'): os.mkdir('model');
        #tf.saved_model.save(self.qnet,'./model/vpg_model');
        self.qnet.save_weights('./model/dqn_model');

def main():

    dqn = DQN();
    dqn.train(1000);

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
