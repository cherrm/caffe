# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:31:20 2015

@author: c
"""
import numpy as np
import os
import datetime
import glob
import itertools
import skimage as ski
import math
import sys
    
# Class that handles image loading, batch preparation and stuff.
class NetDataset:    
    # parameters
    imgFormat = 'png'
    batchSize = 380
    facesPerIdentity = 20
    nNegativeScale = 3.0 # number of negative candidates to chose from

    flipAugmentation = True
    shiftAugmentation = True
    targetSize = None
    printStatus = True
    
    # data
    data = None         # list of image data
    imageIdxRanges = None
    
    # internal attributes
    classPointer = 0
    loops = 0
        
    ###################
    ### public methods
    ###################
    
    # Loads images from disk. Assumes a path/class1/img1.ext structure.
    # return: -
    def loadImageData(self, path):
        
        if (os.path.isfile(path + '/data.npy') and
            os.path.isfile(path + '/classes.npy') and
            os.path.isfile(path + '/imageIdxRanges.npy')):
                
            # found previously saved chunk files, use them because loading is significantly faster this way
            self.data = np.load(path + '/data.npy')
            self.classes = np.load(path + '/classes.npy')
            self.imageIdxRanges = np.load(path + '/imageIdxRanges.npy')
            
        else:
            # build dataset from scratch using the single images
            
            # get all class subfolders
            subdirs = next(os.walk(path))[1]
            
            # read all images and create positive pairs
            if (self.printStatus):
                print('{:s} - Read images and create anchor/positive pairs'.format(str(datetime.datetime.now()).split('.')[0]))

            data = []
            classId = 0
            classes = []

            imageIdxOffset = 0
            imageIdxRanges = []

            for sd in subdirs:
                # get folder name and images
                curDir = path + '/' + sd    
                pData = [self.loadImage(imgName) for imgName in sorted(glob.glob(curDir + '/*.' + self.imgFormat))]  
                
                if len(pData) == 0:
                    continue

                for i in range(len(pData)):
                    imageIdxRanges.append([imageIdxOffset, imageIdxOffset+len(pData)-1])

                elems = range(imageIdxOffset, imageIdxOffset+len(pData))
                np.random.shuffle(elems)
                classes.append(elems)

                imageIdxOffset += len(pData)
    
                # collect data and labels                
                data = data + pData
                            
                # move to next class
                classId += 1
            # shuffle positive pairs for more stable training 
            np.random.shuffle(classes) 
                           
            # save data to class attributes                
            self.data = np.transpose(np.asarray(data), (0,3,1,2))
            self.classes = classes
            self.imageIdxRanges = imageIdxRanges
                
            # save data to disk for faster load on the next call
            np.save('{:s}/data.npy'.format(path), self.data)
            np.save('{:s}/classes.npy'.format(path), self.classes)
            np.save('{:s}/imageIdxRanges.npy'.format(path), self.imageIdxRanges)
        
    # Prepares the next batch for the given data
    # return: (netData, netLabels)
    def getNextClassificationBatch(self):
        # currently still a dummy method     
        netData = np.zeros((32,32))
        netLabels = np.zeros((32,32))
        return (netData, netLabels)
        
    # Prepares the next batch for the given data
    # return: (netData, netLabels)
    def getNextVerficationBatch(self, net):

        currentNAnchorPositives = 0;
        anchorPositives = []

        toRemoveFromNegatives = []

        while (currentNAnchorPositives < self.batchSize):
            clazz = self.classes[self.classPointer]

            # negatives to be removed
            toRemoveFromNegatives.extend(range(self.imageIdxRanges[clazz[0]][0], self.imageIdxRanges[clazz[0]][1]+1))

            elems = []
            if self.facesPerIdentity != 0:
                start = self.loops * self.facesPerIdentity % len(clazz)
                offset = start + self.facesPerIdentity - 1 if start + self.facesPerIdentity - 1 < len(clazz) else start + self.facesPerIdentity - 1 - len(clazz)
                if (start > offset):
                    elems.extend(clazz[start:len(clazz)])
                    elems.extend(clazz[0:offset+1])
                else:
                    if self.facesPerIdentity < len(clazz):
                        elems = clazz[start:offset+1]
                    else:
                        elems = clazz

            anchorPositivesCurrent = np.asarray([p for p in itertools.combinations(elems, 2)], dtype=np.int32)

            anchorPositives.extend(anchorPositivesCurrent)
            currentNAnchorPositives += len(anchorPositivesCurrent)

            if self.classPointer == len(self.classes)-1:
                self.classPointer = 0
                self.loops += 1
            else:
                self.classPointer += 1

        # ensure we have the correct batchsize
        currentNAnchorPositives = self.batchSize
        anchorPositives = anchorPositives[:int(self.batchSize)]

        # get anchors and remove duplicates
        anchors = tuple(x[0] for x in anchorPositives)
        anchors = list(set(anchors))

        negativeCands = [x for x in range(0, len(self.imageIdxRanges)) if x not in toRemoveFromNegatives]  
        np.random.shuffle(negativeCands) 
        nNegatives = math.ceil(len(anchorPositives) * self.nNegativeScale)
        nNegatives = nNegatives if nNegatives < len(negativeCands) else len(negativeCands)
        negatives = negativeCands[:int(nNegatives)] # Todo(ms): crash if we dont have enough negatives, should start from the beginning

        posData3 = self.data[negatives]
        imgData = []
        for i in range(len(negatives)):
            posData1 = np.repeat(self.data[anchorPositives[i][0]], len(negatives))
            posData2 = np.repeat(self.data[anchorPositives[i][1]], len(negatives))
            imgData = np.concatenate(imgData, np.concatenate((posData1, posData2, posData3), axis=1), axis=0)

        # data augmentation
        netData = np.zeros((imgData.shape[0], imgData.shape[1], self.targetSize, self.targetSize), dtype=np.float32)
        for f in range(netData.shape[0]):
            for c in range(netData.shape[1]):
                netData[f,c,:,:] = self.augmentImage(imgData[f,c,:,:])
                        
        # search for (semi) hard negatives and return selected triplets         
        net.set_input_arrays(netData, np.ones((netData), dtype=np.float32))
        net.forward()
        ft1 = net.blobs['feat'].data
        ft2 = net.blobs['feat_p'].data
        ft3 = net.blobs['feat_pp'].data

        distAnchorPositives = np.sum((ft1 - ft2)**2, axis=1) # euclidean distance between anchor and positive
        distAnchorNegatives = np.sum((ft1 - ft3)**2, axis=1) # euclidean distance between anchor and negative
        
        assert len(distAnchorPositives) == len(distAnchorNegatives)
        minDistAnchorNegative = sys.float_info.max
        minDistAnchorNegativeIdx = -1;
        minDistSemiAnchorNegativeFound = False
        netData2 = []
        for i in len(distAnchorPositives):
            if distAnchorNegatives[i] < minDistAnchorNegative:
                if distAnchorPositives[i] < distAnchorNegatives[i]:
                    minDistAnchorNegative = distAnchorNegatives[i]
                    minDistAnchorNegativeIdx = i
                    minDistSemiAnchorNegativeFound = True
                else if !minDistSemiAnchorNegativeFound:
                    minDistAnchorNegative = distAnchorNegatives[i]
                    minDistAnchorNegativeIdx = i

            if i != 0 && i % len(negatives) == 0:
                if minDistAnchorNegativeIdx != -1:
                    netData2 = np.concatenate(netData2, netData[minDistAnchorNegativeIdx])
                    minDistAnchorNegative = sys.float_info.max
                    minDistAnchorNegativeIdx = -1; 
                    minDistSemiAnchorNegativeFound = False 

        return (netData2, np.ones((netData2), dtype=np.float32))   

    #####################    
    ### private methods
    #####################
        
    # load image, convert to grayscale and scale to [0,1]
    # return: img
    def loadImage(self, filename):        
        # read image
        img = ski.io.imread(filename)
        # convert to grayscale if needed
        if (img.ndim >= 3 and img.shape[2] > 1):
            img = ski.color.rgb2gray(img)
        elif (img.ndim == 2):
            img = img[:,:,np.newaxis]
            
        # convert to float datatype
        img = ski.img_as_float(img).astype(np.float32)
        return img
        
    # Method for dataset augmentation. Performs random flipping and shifting
    # return: image 
    def augmentImage(self, image):
        assert((self.targetSize is not None) or not self.shiftAugmentation)
        # perform flipping
        if (self.flipAugmentation and np.random.randint(0,2) < 1):
            image = np.fliplr(image)
            
        # perform shifting
        padSize = image.shape[0] - self.targetSize;
        if (self.shiftAugmentation and padSize > 0):
            # random shift + cut
            offsetX = np.random.randint(0, padSize+1);    # random number between 0 and padSize
            offsetY = np.random.randint(0, padSize+1);    # random number between 0 and padSize
            image = image[offsetY:self.targetSize+offsetY,offsetX:self.targetSize+offsetX];
        elif (padSize > 0):
            # center cut
            offset = np.rint(padSize/2)
            image = image[offset:self.targetSize+offset,offset:self.targetSize+offset];
            
        return image        