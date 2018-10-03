#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import;
from __future__ import division;
from __future__ import print_function;
from __future__ import unicode_literals;

import math;
import random;
import numpy as np;
import cv2;
from caffe2.python import core, workspace, model_helper, utils, brew;
from caffe2.proto import caffe2_pb2;
from caffe2.python.optimizer import build_sgd;
from ale_python_interface import ALEInterface;

class PPO(object):
        # class members
        #
        # predict_net: predict network
        # model: init + forward + backward + update network model helper
        # gamma: reward discount
        # batch_size: batch number
        # policy_layers: perceptron numbers of policy subnetwork layers
        # value_layers: perceptron numbers of value subnetwork layers
        #
        # ale: atari learning environment object
        # legal_actions: a map from index to action number
        def __init__(self):
                self.ale = ALEInterface();
                self.ale.loadROM(str.encode('Boxing.bin'));
                self.legal_actions = self.ale.getMinimalActionSet();
                self.gamma = 0.8;
                self.batch_size = 5120;
                self.policy_layers = [200,100];
                self.value_layers = [200,100];

        def CreateModel(self):
                model = model_helper.ModelHelper(name = "model");
                #dimension of data is batch x 4 x 84 x 84
                #dimension of label is batch x 18
                data = np.zeros([self.batch_size,4,84,84],dtype = np.float32);
                policy_label = np.zeros([self.batch_size,self.legal_actions.shape[0]],dtype = np.float32);
                value_label = np.zeros([self.batch_size,1],dtype = np.float32);
                workspace.FeedBlob('data',data);
                workspace.FeedBlob('policy_label',policy_label);
                workspace.FeedBlob('value_label',value_label);
                data,policy_label,value_label = model.net.AddExternalInputs('data','policy_label','value_label');
                #dimension of label is batch x 28224
                flat = model.net.Flatten([data],'flat');
                #policy subnet
                for idx,dim_out in enumerate(self.policy_layers):
                        if idx == 0:
                                tensor_in = flat;
                                dim_in = 28224;
                        else:
                                tensor_in = 'policy_conv' + str(idx);
                                dim_in = self.policy_layers[idx - 1];
                        globals()['policy_conv' + str(idx + 1)] = brew.fc(model, tensor_in, 'policy_conv' + str(idx + 1), dim_in = dim_in, dim_out = dim_out);
                        globals()['policy_conv' + str(idx + 1)] = brew.relu(model, 'policy_conv' + str(idx + 1), 'policy_conv' + str(idx + 1));
                logits = brew.fc(model, 'policy_conv' + str(len(self.policy_layers)), 'logits', dim_in = self.policy_layers[-1], dim_out = len(self.legal_actions));
                #value subnet
                for idx,dim_out in enumerate(self.value_layers):
                        if idx == 0:
                                tensor_in = flat;
                                dim_in = 28224;
                        else:
                                tensor_in = 'value_conv' + str(idx);
                                dim_in = self.value_layers[idx - 1];
                        globals()['value_conv' + str(idx + 1)] = brew.fc(model, tensor_in, 'value_conv' + str(idx + 1), dim_in = dim_in, dim_out = dim_out);
                        globals()['value_conv' + str(idx + 1)] = brew.relu(model, 'value_conv' + str(idx + 1), 'value_conv' + str(idx + 1));
                #value
                value = brew.fc(model, 'value_conv' + str(len(self.value_layers)), 'value', dim_in = self.value_layers[-1], dim_out = 1);
                #policy
                softmax = model.net.Softmax(logits,'softmax', axis = 1);
                #predict network
                self.predict_net = core.Net(model.net.Proto());
                #loss
                train_value = np.zeros([1],dtype = np.float32);
                train_policy = np.zeros([1],dtype = np.float32);
                workspace.FeedBlob('train_value',train_value);
                workspace.FeedBlob('train_policy',train_policy);
                train_value,train_policy = model.net.AddExternalInputs('train_value','train_policy');

                mul = model.net.Mul([softmax,policy_label],'mul');
                ip = model.net.ReduceSum(mul,'ip',axes = [1]);
                xent = model.net.Negative(ip,'xent');
                loss1 = model.net.ReduceMean(xent, 'loss1', axes = [0]);
                policy_loss = model.net.Mul([loss1,train_policy],'policy_loss',broadcast = 1);
                model.net.Print(policy_loss,[],to_file = 0);

                sq2d = model.net.SquaredL2Distance([value,value_label],'sq2d');
                loss2 = model.net.ReduceMean(sq2d, 'loss2', axes = [0]);
                value_loss = model.net.Mul([loss2,train_value],'value_loss',broadcast = 1);
                model.net.Print(value_loss,[],to_file = 0);
                #gradient
                model.AddGradientOperators([policy_loss,value_loss]);
                #sgd
                build_sgd(model, base_learning_rate = 1e-7, policy = 'step', stepsize = 1000, gamma = 0.9999);

                self.model = model;

                #instantiate networks
                workspace.RunNetOnce(self.model.param_init_net);
                workspace.CreateNet(self.predict_net);
                workspace.CreateNet(self.model.net);

        def PlayOneEpisode(self):
                self.ale.reset_game();
                transforms = [];
                #stuff frame buffer with 4 frames
                frames = [];
                reward_total = 0;
                timestamp = 0;
                #initial status
                for i in range(4):
                        current_frame = cv2.resize(self.ale.getScreenGrayscale(),(84,84));
                        frames.append(current_frame);
                        assert False == self.ale.game_over();
                #player until game over
                while False == self.ale.game_over():
                        #choose action
                        action = np.random.choice(self.legal_actions);
                        #apply action
                        reward = 0;
                        for i in range(3):
                                reward = reward + self.ale.act(action);
                                if self.ale.game_over(): break
                        #get new observation
                        current_frame = cv2.resize(self.ale.getScreenGrayscale(),(84,84));
                        #save transformation
                        transforms.append({"status":frames,"action_index":np.where(self.legal_actions == action)[0][0],"reward":reward,"new_frame":current_frame});
                        #update status
                        frames.append(current_frame);
                        frames.pop(0);
                        #update total reward
                        reward_total = reward_total + math.pow(self.gamma,timestamp) * reward;
                        timestamp = timestamp + 1;
                for ts in transforms:
                        ts["reward_total"] = reward_total;
                return transforms;

        def Value(self,data):
                assert data.shape == (4,84,84);
                data = np.expand_dims(data,axis = 0);
                workspace.FeedBlob("data",data);
                workspace.RunNet(self.predict_net.Name());
                return workspace.FetchBlob("value");

        def Policy(self,data):
                assert data.shape == (4,84,84);
                data = np.expand_dims(data,axis = 0);
                workspace.FeedBlob("data",data);
                workspace.RunNet(self.predict_net.Name());
                return workspace.FetchBlob("softmax");

        def TrainModel(self):
                #start training
                while True:
                        dataset = [];
                        print("1) sample 1000 episodes");
                        for i in range(300):
                                print("sampling ",i+1," episode");
                                dataset = dataset + self.PlayOneEpisode();
                        random.shuffle(dataset);
                        print("2) update value network");
                        for i in range(1000):
                                batch = np.random.choice(dataset,self.batch_size,replace = False);
                                data = np.zeros([self.batch_size,4,84,84],dtype = np.float32);
                                label = np.zeros([self.batch_size,1],dtype = np.float32);
                                for j in range(self.batch_size):        
                                        data[j,...] = np.array(batch[j]["status"]);
                                        label[j,0] = batch[j]["reward_total"];
                                workspace.FeedBlob("data",data);
                                workspace.FeedBlob("value_label",label);
                                workspace.FeedBlob("train_value",np.array([1],dtype = np.float32));
                                workspace.FeedBlob("train_policy",np.array([0],dtype = np.float32));
                                workspace.RunNet(self.model.name);
                        print("3) update policy network");
                        for i in range(1000):
                                batch = np.random.choice(dataset,self.batch_size,replace = False);
                                data = np.zeros([self.batch_size,4,84,84],dtype = np.float32);
                                label = np.zeros([self.batch_size,self.legal_actions.shape[0]],dtype = np.float32);
                                for j in range(self.batch_size):
                                        data[j,...] = np.array(batch[j]["status"]);
                                        next_status = batch[j]["status"];
                                        next_status.append(batch[j]["new_frame"]);
                                        next_status.pop(0);
                                        current_value = self.Value(np.array(batch[j]["status"],dtype = np.float32));
                                        next_value = self.Value(np.array(next_status,dtype = np.float32));
                                        advantage = - current_value + batch[j]["reward"] + self.gamma * next_value;
                                        label[j,batch[j]["action_index"]] = advantage / (1.0/self.legal_actions.shape[0]);
                                workspace.FeedBlob("data",data);
                                workspace.FeedBlob("policy_label",label);
                                workspace.FeedBlob("train_value",np.array([0],dtype = np.float32));
                                workspace.FeedBlob("train_policy",np.array([1],dtype = np.float32));
                                workspace.RunNet(self.model.name);
                        #save model
                        self.SaveModel('model/init.pb','model/predict.pb');

        def SaveModel(self,init_path,pred_path):
                #output predict net
                with open(pred_path,'wb') as f:
                        f.write(self.model.net._net.SerializeToString());
                #output init net
                init_net = caffe2_pb2.NetDef();
                for param in self.model.params:
                        blob = workspace.FetchBlob(param);
                        shape = blob.shape;
                        op = core.CreateOperator('GivenTensorFill',[],[param],arg = [utils.MakeArgument('shape',shape),utils.MakeArgument('values',blob)]);
                        init_net.op.extend([op]);
                init_net.op.extend([core.CreateOperator('ConstantFill',[],['data'],shape = (self.batch_size,4,84,84))]);
                init_net.op.extend([core.CreateOperator('ConstantFill',[],['policy_label'],shape = (self.batch_size,self.legal_actions.shape[0]))]);
                init_net.op.extend([core.CreateOperator('ConstantFill',[],['value_label'],shape = (self.batch_size,1))]);
                with open(init_path,'wb') as f:
                        f.write(init_net.SerializeToString());

def main():
        device = core.DeviceOption(caffe2_pb2.CUDA);
        with core.DeviceScope(device):
                model = PPO();
                model.CreateModel();
                model.TrainModel();

if __name__ == '__main__':
        main();

