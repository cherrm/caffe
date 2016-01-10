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
    
# Class that handles image loading, batch preparation and stuff.
class NetDataset:    

    def __init__(self):
        # parameters
        self.imgFormat = 'png'
        self.batchSize = 380
        self.facesPerIdentity = 20

        self.flipAugmentation = True
        self.shiftAugmentation = True
        self.targetSize = None
        self.printStatus = True
    
        # data
        self.data = None         # list of image data
        self.posData1 = None
        self.posData2 = None
        self.posData3 = None  

        self.epoch = 0
        self.batchCount = 0

    ###################
    ### public methods
    ###################
    
    # Loads images from disk. Assumes a path/class1/img1.ext structure.
    # return: -
    def loadImageData(self, path):
                
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

        assert (self.batchSize <= len(data)) 
        self.data = np.transpose(np.asarray(data), (0,3,1,2))
            
        currentNAnchorPositives = 0
        anchorPositives = []
        toRemoveFromNegatives = []
        foundInLoop = False
        fill = False
        run = True
        classPointer = 0
        classOfRest = 0
        rest = [] # rest of anchor positive pairs in set of combinations not used in current loop
        loops = 0

        posData1 = []
        posData2 = []
        posData3 = []  

        while run:
            while (currentNAnchorPositives < self.batchSize):

                start = -1

                if len(rest) != 0:
                    clazz = classes[classOfRest]
                    if self.batchSize-currentNAnchorPositives >= len(rest):
                        anchorPositives.extend(rest)
                        rest = []
                        classOfRest = 0
                    else:
                        anchorPositives.extend(rest[:int(self.batchSize-currentNAnchorPositives)])
                        del rest[int(self.batchSize-currentNAnchorPositives):]
                    currentNAnchorPositives += len(anchorPositives)
                    toRemoveFromNegatives.extend(range(imageIdxRanges[clazz[0]][0], imageIdxRanges[clazz[0]][1]+1))

                if currentNAnchorPositives < self.batchSize:
                    elems = []
                    clazz = classes[classPointer]
                    if self.facesPerIdentity != 0:
                        start = loops * self.facesPerIdentity if loops * self.facesPerIdentity < len(clazz) else -1
                        if start != -1:
                            foundInLoop = True
                            offset = self.facesPerIdentity if start + self.facesPerIdentity-1 < len(clazz) else len(clazz) - start
                            elems = clazz[start:start+offset]
                            if len(elems) == 1 and len(clazz) > 1:
                                elems.extend(clazz[:1])
                            anchorPositivesCurrent = np.asarray([p for p in itertools.combinations(elems, 2)], dtype=np.int32)
                            if len(anchorPositivesCurrent) + currentNAnchorPositives > self.batchSize:
                                anchorPositives.extend(anchorPositivesCurrent[:int(self.batchSize-currentNAnchorPositives)])
                                rest.extend(anchorPositivesCurrent[int(self.batchSize-currentNAnchorPositives):])
                                classOfRest = classPointer
                                currentNAnchorPositives = self.batchSize
                            else:
                                anchorPositives.extend(anchorPositivesCurrent)
                                currentNAnchorPositives += len(anchorPositivesCurrent)
                            #setup array with negatives
                            toRemoveFromNegatives.extend(range(imageIdxRanges[clazz[0]][0], imageIdxRanges[clazz[0]][1]+1))

                if classPointer == len(classes)-1:
                    classPointer = 0
                    if not foundInLoop:
                        fill = True
                        loops = 0
                    else:
                        loops += 1
                    foundInLoop = False
                else:
                    classPointer += 1

            # ensure we have the correct batchsize
            assert currentNAnchorPositives == self.batchSize

            negativeCands = [x for x in range(0, len(imageIdxRanges)) if x not in toRemoveFromNegatives]  

            # fill negativeCands
            if len(negativeCands) > 0 and len(anchorPositives) > len(negativeCands):
                negativeCands = np.repeat(negativeCands, math.ceil(float(len(anchorPositives)) / len(negativeCands)))

            if len(anchorPositives) <= len(negativeCands):
                np.random.shuffle(negativeCands) 
                negativeSelection = negativeCands[:int(len(anchorPositives))]
                posData1.extend([x[0] for x in anchorPositives])
                posData2.extend([x[1] for x in anchorPositives])
                posData3.extend(negativeSelection)
            else:
                print "not enough negatives for current batch, skipping..."

            currentNAnchorPositives = 0
            anchorPositives = []
            toRemoveFromNegatives = []

            if fill:
                run = False

        assert (len(posData1) == len(posData2))
        assert (len(posData1) == len(posData3))

        self.posData1 = posData1;
        self.posData2 = posData2;
        self.posData3 = posData3;

    # Prepares the next batch for the given data
    # return: (netData, netLabels)
    def getNextClassificationBatch(self):
        # currently still a dummy method     
        netData = np.zeros((32,32))
        netLabels = np.zeros((32,32))
        return (netData, netLabels)
        
    # Prepares the next batch for the given data
    # return: (netData, netLabels)
    def getNextVerficationBatch(self):

        start = self.batchCount * self.batchSize
        if start+self.batchSize >= len(self.posData1):
            start = 0
            self.epoch += 1
            self.batchCount = 0
        else:
            self.batchCount += 1

        posData1 = self.data[self.posData1[start:start+self.batchSize]]
        posData2 = self.data[self.posData2[start:start+self.batchSize]]
        posData3 = self.data[self.posData3[start:start+self.batchSize]]

        # combine data
        imgData = np.concatenate((posData1, posData2, posData3), axis=1)
            
        # data augmentation
        netData = np.zeros((imgData.shape[0], imgData.shape[1], self.targetSize, self.targetSize), dtype=np.float32)
        for f in range(netData.shape[0]):
            for c in range(netData.shape[1]):
                netData[f,c,:,:] = self.augmentImage(imgData[f,c,:,:])
                        
        if (self.printStatus):
            print('{:s} - Starting epoch {:d}'.format(str(datetime.datetime.now()).split('.')[0], self.epoch))

        assert len(netData) == self.batchSize
        return (netData, np.ones((self.batchSize), dtype=np.float32))   

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