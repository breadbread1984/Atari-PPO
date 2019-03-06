#!/usr/bin/python3

import os;
import random;
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
    
    MEMORY_LIMIT = 900000;
    BATCH_SIZE  = 128;
    BURNIN_STEP = 50000;
    TRAIN_FREQUENCY = 4;
    UPDATE_FREQUENCY = 40000;
    STATUS_SIZE = 4;
    GAMMA = 1;
    
    def __init__(self):
        
        # ale related members
        self.ale = ALEInterface();
        self.ale.loadROM(get_game_path('boxing'));
        self.legal_actions = self.ale.getMinimalActionSet();
        self.status = list();
        # use qnet_latest to hold the latest updated weights 
        self.qnet_latest = QNet(self.legal_actions);
        # use qnet_target to hold the target model weights
        self.qnet_target = QNet(self.legal_actions);
        if True == os.path.exists('model'):
            self.qnet_latest.load_weights('./model/dqn_model');
        # use qnet_target as the rollout model
        self.qnet_target.set_weights(self.qnet_latest.get_weights());
        # status transition memory
        self.memory = list();

    def convertImgToTensor(self,status):
        
        # status.shape = [4,48,48,1]
        status = tf.convert_to_tensor(status, dtype = tf.float32);
        status = tf.transpose(status,[1,2,0]);
        status = tf.expand_dims(status,0);
        return status;
    
    def convertBatchToTensor(self,batch):
        
        st,at,rt,stp1,et = zip(*batch);
        # st.shape = [batchsize,48,48,4]
        st = tf.convert_to_tensor(st, dtype = tf.float32);
        at = tf.convert_to_tensor(at, dtype = tf.int32);
        rt = tf.convert_to_tensor(rt, dtype = tf.float32);
        stp1 = tf.convert_to_tensor(stp1, dtype = tf.float32);
        et = tf.convert_to_tensor(et, dtype = tf.bool);
        return (st,at,rt,stp1,et);
    
    def getObservation(self):
        
        image = self.ale.getScreenGrayscale();
        frame = image[25:185,:,:];
        frame = cv2.resize(frame,(84,84)) / 255.0;
        return frame;
    
    def remember(self, transition):
        
        if len(self.memory) > self.MEMORY_LIMIT: self.memory.pop(0);
        self.memory.append(transition);
    
    def reset_game(self):
        
        self.ale.reset_game();
        for i in range(self.STATUS_SIZE):
            current_frame = self.getObservation();
            self.status.append(current_frame);
            assert False == self.ale.game_over();

    def rollout(self):
        
        if self.ale.game_over(): self.reset_game();
        # display screen
        cv2.imshow('screen',self.ale.getScreenRGB());
        cv2.waitKey(1);
        # choose action 
        st = self.convertImgToTensor(self.status);
        Qt = self.qnet_target(st);
        action_index = tf.random.categorical(tf.math.exp(Qt),1);
        reward = 0;
        for i in range(self.STATUS_SIZE):
            reward += self.ale.act(self.legal_actions[action_index]);
        self.status.append(self.getObservation());
        self.status.pop(0);
        stp1 = self.convertImgToTensor(self.status);
        game_over = self.ale.game_over();
        self.remember((st,action_index,reward,stp1,game_over));
        return reward;
    
    def train(self, loop_time = 10000000):
        
        optimizer = tf.keras.optimizers.Adam(1e-3);
        # setup checkpoint and log utils
        checkpoint = tf.train.Checkpoint(model = self.qnet_target, optimizer = optimizer, optimizer_step = optimizer.iterations);
        checkpoint.restore(tf.train.latest_checkpoint('checkpoints_dqn'));
        log = tf.summary.create_file_writer('checkpoints_dqn');
        self.reset_game();
        for i in range(loop_time):
            self.rollout();
            # do nothing if collected samples are not enough
            if i < self.BURNIN_STEP or len(self.memory) < self.BATCH_SIZE:
                continue;
            # update qnet_latest at certain frequency
            if i % self.TRAIN_FREQUENCY == 0:
                avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
                # random sample from memory
                batch = random.sample(self.memory, self.BATCH_SIZE);
                st,at,rt,stp1,et = self.convertBatchToTensor(batch);
                # policy loss
                with tf.GradientTape() as tape:
                    Qt = self.qnet_target(st);
                    Qtp1 = self.qnet_target(stp1);
                    action_mask = tf.one_hot(at,len(self.legal_actions));
                    qt = tf.math.reduce_sum(action_mask * Qt, axis = 1);
                    qtp1 = tf.math.reduce_max(Qtp1, axis = 1);
                    value = rt + tf.cond(tf.equal(et,False),lambda:self.GAMMA * qtp1,lambda:0);
                    loss = tf.math.squared_difference(qt, value);
                    avg_loss.update_state(loss);
                # write loss to summary
                if tf.equal(optimizer.iterations % 100, 0):
                    with log.as_default():
                        tf.summary.scalar('loss',avg_loss.result(), step = optimizer.iterations);
                    avg_loss.reset_states();
                # train qnet_latest
                grads = tape.gradient(loss,self.qnet_latest.variables);
                optimizer.apply_gradients(zip(grads,self.qnet_latest.variables));
            # save model every episode
            if i % self.UPDATE_FREQUENCY == 0:
                self.qnet_target.set_weights(self.qnet_latest.get_weights());
                checkpoint.save(os.path.join('checkpoints_dqn','ckpt'));
        # save final model
        if False == os.path.exists('model'): os.mkdir('model');
        #tf.saved_model.save(self.qnet,'./model/vpg_model');
        self.qnet_target.save_weights('./model/dqn_model');

def main():

    dqn = DQN();
    dqn.train();

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
