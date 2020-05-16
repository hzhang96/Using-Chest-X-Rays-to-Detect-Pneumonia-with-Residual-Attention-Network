import os
import numpy as np
import time
import sys
# -*- coding: utf-8 -*
from ChexnetTrainer import ChexnetTrainer

#-------------------------------------------------------------------------------- 

def main ():
    
    runTest()
    #runTrain()
  
#--------------------------------------------------------------------------------   

def runTrain():
    
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    RESIDUAL    = 'Residual-attention'
    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime
    
    #---- Path to the directory with images
    pathDirData = './database'
    
    #---- Paths to the files with training, validation and testing sets.
    #---- Each file should contains pairs [path to image, output vector]
    #---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = './dataset/train_1.txt'
    pathFileVal = './dataset/val_1.txt'
    pathFileTest = './dataset/test_1.txt'
    
    #---- Neural network parameters: type of the network, is it pre-trained 
    #---- on imagenet, number of classes
    nnArchitecture = RESIDUAL
    nnIsTrained = True
    nnClassCount = 14
    
    #---- Training settings: batch size, maximum number of epochs

    trBatchSize = 16

    trMaxEpoch = 20
    
    #---- Parameters related to image transforms: size of the down-scaled image, cropped image
    #
    imgtransResize = 256
    imgtransCrop = 224
        
    #pathModel = 'm-' + timestampLaunch + '.pth.tar'
#layer_56_m-01052020-092834.pth.tar
    pathModel = 'layer_56_m-01052020-161549.pth.tar' #None   #'layer_56_m-01052020-092834.pth.tar'    

    print ('Training NN architecture = ', nnArchitecture)
    ChexnetTrainer.train(pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, pathModel)
    
    print ('Testing the trained model')
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

def runTest():
    
    pathDirData = './database'
    pathFileTest = './dataset/test_1.txt'
    nnArchitecture = 'Residual-attention'
    nnIsTrained = True
    nnClassCount = 14
    trBatchSize = 1
    #---- used to be  of the two numbers
    imgtransResize = 256
    imgtransCrop = 224
    
    pathModel = 'layer_92_m-03052020-021123.pth.tac'
    
    timestampLaunch = ''
    
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize, imgtransCrop, timestampLaunch)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()





