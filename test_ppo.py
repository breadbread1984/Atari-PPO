#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import;
from __future__ import division;
from __future__ import print_function;
from __future__ import unicode_literals;

import numpy as np;
import cv2;
from caffe2.python import core, workspace, model_helper, utils, brew;
from caffe2.proto import caffe2_pb2;
from ale_python_interface import ALEInterface;

def main():
        #load trained model
        with open("model/init.pb",mode = 'rb') as f: 
                init_net = f.read();
        with open("model/predict.pb",mode = 'rb') as f: 
                predict_net = f.read();
        p = workspace.Predictor(init_net,predict_net);
        #load game ROM
        ale = ALEInterface();
        ale.loadROM(str.encode("Boxing.bin"));
        legal_actions = ale.getMinimalActionSet();
        #start to play
        while True:
                frames = [];
                for i in range(4):
                        current_frame = cv2.resize(ale.getScreenGrayscale(),(84,84));
                        frames.append(current_frame);
                        assert False == ale.game_over();
                while False == ale.game_over():
                        current_frame = cv2.resize(ale.getScreenGrayscale(),(84,84));
                        frames.append(current_frame);
                        frames.pop(0);
                        data = np.array(frames);
                        data = np.expand_dims(data,axis = 0);
                        [value,softmax] = p.run({"data":data},["value","softmax"]);
                        #apply actiono
                        for i in range(3): ale.act(legal_actions[np.softmax(softmax,axis = 1)][0]);
                        #show status
                        show = ale.getScreenRGB();
                        cv2.imshow("game",show);
                        cv2.waitKey(20);
                ale.reset_game();

if __name__ == "__main__":
        main();

