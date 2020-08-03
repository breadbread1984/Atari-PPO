#!/usr/bin/python3

import os;
import random;
import numpy as np;
import cv2;
import tensorflow as tf;
from atari_py import ALEInterface;
from atari_py import get_game_path;

def QNet(action_num, hidden_sizes = [32, 20]):

  inputs = tf.keras.Input((84, 84, 4)); # inputs.shape = (batch, height, width, 4)
  results = tf.keras.layers.Flatten()(inputs); # results.shape = (batch, height * width);
  for size in hidden_sizes:
    results = tf.keras.layers.Dense(units = size)(results); # results.shape = (batch, units);
  results = tf.keras.layers.Dense(units = action_num)(results); # results.shape = (batch, action_num)
  return tf.keras.Model(inputs = inputs, outputs = results);

def Loss(action_num, gamma):

  Qt = tf.keras.Input((action_num, ), dtype = tf.float32); # Qt.shape = (batch, action_num)
  Qtp1 = tf.keras.Input((action_num, ), dtype = tf.float32); # Qtp1.shape = (batch, action_num)
  rt = tf.keras.Input((), dtype = tf.float32); # rt.shape = (batch,)
  at = tf.keras.Input((), dtype = tf.int32); # at.shape = (batch,)
  et = tf.keras.Input((), dtype = tf.float32); # et.shape = (batch,)
  action_mask = tf.keras.layers.Lambda(lambda x, n: tf.one_hot(x, n), arguments = {'n': action_num})(at); # action_mask.shape = (batch, action_num)
  qt = tf.keras.layers.Lambda(lambda x: tf.math.reduce_sum(x[0] * x[1], axis = 1))([action_mask, Qt]); # qt.shape = (batch,)
  qtp1 = tf.keras.layers.Lambda(lambda x: tf.math.reduce_max(x, axis = 1))(Qtp1); # max_a Q(s, a).shape = (batch,)
  value = tf.keras.layers.Lambda(lambda x, g: x[0] + g * (tf.ones_like(x[1]) - x[1]) * x[2], arguments = {'g': gamma})([rt, et, qtp1]); # value.shape = (batch,)
  loss = tf.keras.losses.MSE(value, qt);
  return tf.keras.Model(inputs = (Qt, Qtp1, rt, at, et), outputs = loss);

class DQN(object):
    
    SHOW = False;
    SCALE = 10000;
    MEMORY_LIMIT = 4 * SCALE;
    BATCH_SIZE  = 32;
    BURNIN_STEP = 5 * SCALE;
    TRAIN_FREQUENCY = 4;
    UPDATE_FREQUENCY = SCALE;
    STATUS_SIZE = 4;
    GAMMA = 0.99;
    
    def __init__(self):
        
        # ale related members
        self.ale = ALEInterface();
        self.ale.loadROM(get_game_path('boxing'));
        self.legal_actions = self.ale.getMinimalActionSet();
        self.status = list();
        # use qnet_latest to hold the latest updated weights 
        self.qnet_latest = QNet(len(self.legal_actions));
        # use qnet_target to hold the target model weights
        self.qnet_target = QNet(len(self.legal_actions));
        if True == os.path.exists('model'):
            self.qnet_latest.load_weights('./model/dqn_model');
        # use qnet_target as the rollout model
        self.qnet_target.set_weights(self.qnet_latest.get_weights());
        # loss
        self.loss = Loss(len(self.legal_actions), self.GAMMA);
        # status transition memory
        self.memory = list();

    def convertImgToTensor(self,status):
        
        status = tf.constant(status, dtype = tf.float32); # status.shape = (4, 48, 48)
        status = tf.transpose(status, (1, 2, 0)); # status.shape = (48, 48, 4)
        status = tf.expand_dims(status, axis = 0); # status.shape = (1, 48, 48, 4)
        return status;
    
    def convertBatchToTensor(self,batch):
        
        st, at, rt, stp1, et = zip(*batch);
        # st.shape = batchsize*[1,48,48,4]
        st = tf.squeeze(tf.concat(st, axis = 0));
        at = tf.squeeze(tf.concat(at, axis = 0));
        rt = tf.squeeze(tf.concat(rt, axis = 0));
        stp1 = tf.squeeze(tf.concat(stp1, axis = 0));
        et = tf.squeeze(tf.concat(et, axis = 0));
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
        self.status = list();
        for i in range(self.STATUS_SIZE):
            current_frame = self.getObservation();
            self.status.append(current_frame);
            assert False == self.ale.game_over();

    def rollout(self):
        
        if self.ale.game_over() or len(self.status) != self.STATUS_SIZE:
            self.reset_game();
        if self.SHOW:
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
        game_over = 1. if self.ale.game_over() else 0.;
        self.remember((st, action_index, float(reward), stp1, game_over));
        return game_over;
    
    def train(self, loop_time = 10000000):
        
        optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(0.00025, 5 * self.SCALE, 0.96));
        # setup checkpoint and log utils
        checkpoint = tf.train.Checkpoint(model = self.qnet_target, optimizer = optimizer, optimizer_step = optimizer.iterations);
        checkpoint.restore(tf.train.latest_checkpoint('checkpoints_dqn'));
        log = tf.summary.create_file_writer('checkpoints_dqn');
        avg_reward = tf.keras.metrics.Mean(name = 'reward', dtype = tf.float32);
        self.reset_game();
        for i in range(loop_time):
            game_over = self.rollout();
            # do nothing if collected samples are not enough
            if i < self.BURNIN_STEP or len(self.memory) < self.BATCH_SIZE:
                continue;
            # update qnet_latest at certain frequency
            if i % self.TRAIN_FREQUENCY == 0:
                avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
                # random sample from memory
                batch = random.sample(self.memory, self.BATCH_SIZE);
                st, at, rt, stp1, et = self.convertBatchToTensor(batch);
                # policy loss
                with tf.GradientTape() as tape:
                    Qt = self.qnet_latest(st);
                    Qtp1 = self.qnet_latest(stp1);
                    loss = self.loss([Qt, Qtp1, rt, at, et]);
                    avg_loss.update_state(loss);
                # write loss to summary
                if tf.equal(optimizer.iterations % 100, 0):
                    with log.as_default():
                        tf.summary.scalar('loss',avg_loss.result(), step = optimizer.iterations);
                    avg_loss.reset_states();
                # train qnet_latest
                grads = tape.gradient(loss,self.qnet_latest.trainable_variables);
                optimizer.apply_gradients(zip(grads,self.qnet_latest.trainable_variables));
            # save model every episode
            if i % self.UPDATE_FREQUENCY == 0:
                self.qnet_target.set_weights(self.qnet_latest.get_weights());
                checkpoint.save(os.path.join('checkpoints_dqn','ckpt'));
                # evaluate the updated model
                for i in range(10): avg_reward.update_state(self.eval(steps = 200));
                with log.as_default():
                    tf.summary.scalar('reward', avg_reward.result(), step = optimizer.iterations);
                print('Step #%d Reward: %.6f lr: %.6f' % (optimizer.iterations, avg_reward.result(), optimizer._hyper['learning_rate'](optimizer.iterations)));
                avg_reward.reset_states();
        # save final model
        if False == os.path.exists('model'): os.mkdir('model');
        #tf.saved_model.save(self.qnet,'./model/vpg_model');
        self.qnet_target.save_weights('./model/dqn_model');
        
    def eval(self, steps = None):
        self.ale.reset_game();
        status = list();
        # full initial status
        for i in range(self.STATUS_SIZE):
            current_frame = self.getObservation();
            status.append(current_frame);
            assert False == self.ale.game_over();
        # play one episode
        total_reward = 0;
        step = 0;
        while False == self.ale.game_over() and (steps is None or step < steps):
            if self.SHOW:
                # display screen
                cv2.imshow('screen',self.ale.getScreenRGB());
                cv2.waitKey(1);
            st = self.convertImgToTensor(status);
            Qt = self.qnet_latest(st);
            action_index = tf.random.categorical(tf.math.exp(Qt),1);
            for i in range(self.STATUS_SIZE):
                total_reward += self.ale.act(self.legal_actions[action_index]);
            status.append(self.getObservation());
            status.pop(0);
            step += 1;
        return total_reward;

def main():

    dqn = DQN();
    dqn.train();

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();
