import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim
import random, os
import numpy as np

def ComputeMean(imagesList, depth=256):
    r,g,b,i = 0.0, 0.0, 0.0, 0.0
    for img in imagesList:
        try:
            rImg, gImg, bImg = img.split()
            r+=np.mean(np.array(rImg))
            g+=np.mean(np.array(gImg))
            b+=np.mean(np.array(bImg))
            i+=1
        except:
            pass
    return r/i/depth, g/i/depth, b/i/depth

def ComputeStdDev(imagesList, depth=256):
    r,g,b,i = 0.0, 0.0, 0.0, 0.0
    for img in imagesList:
        try:
            rImg, gImg, bImg = img.split()
            r+=np.std(np.array(rImg))
            g+=np.std(np.array(gImg))
            b+=np.std(np.array(bImg))
            i+=1
        except:
            pass
    return r/i/depth, g/i/depth, b/i/depth

def copyFeaturesParametersAlexnet(net, netBase):
    for i, f in enumerate(net.features):
        if type(f) is torch.nn.modules.conv.Conv2d:
            print "copy", f
            f.weight.data = netBase.features[i].weight.data
            f.bias.data = netBase.features[i].bias.data

def copysClassifierParameterAlexnet(net, netBase):
    for i, c in enumerate(net.classifier):
        if type(c) is torch.nn.modules.linear.Linear :
            if c.weight.size() == netBase.classifier[i].weight.size():
                print "copy", c
                c.weight.data = netBase.classifier[i].weight.data
                c.bias.data = netBase.classifier[i].bias.data 


def copyFeaturesParametersResnet50(net, netBase):
    net.conv1.weight.data = netBase.conv1.weight.data
    net.bn1.weight.data = netBase.bn1.weight.data
    net.bn1.bias.data = netBase.bn1.bias.data

    lLayer = [(net.layer1, netBase.layer1, 3),
              (net.layer2, netBase.layer2, 4),
              (net.layer3, netBase.layer3, 6),
              (net.layer4, netBase.layer4, 3)
             ]

    for targetLayer, rootLayer, nbC in lLayer:
        for i in range(nbC):
            targetLayer[i].conv1.weight.data = rootLayer[i].conv1.weight.data
            targetLayer[i].bn1.weight.data = rootLayer[i].bn1.weight.data
            targetLayer[i].bn1.bias.data = rootLayer[i].bn1.bias.data
            targetLayer[i].conv2.weight.data = rootLayer[i].conv2.weight.data
            targetLayer[i].bn2.weight.data = rootLayer[i].bn2.weight.data
            targetLayer[i].bn2.bias.data = rootLayer[i].bn2.bias.data
            targetLayer[i].conv3.weight.data = rootLayer[i].conv3.weight.data
            targetLayer[i].bn3.weight.data = rootLayer[i].bn3.weight.data
            targetLayer[i].bn3.bias.data = rootLayer[i].bn3.bias.data
        targetLayer[0].downsample[0].weight.data = rootLayer[0].downsample[0].weight.data
        targetLayer[0].downsample[1].weight.data = rootLayer[0].downsample[1].weight.data
        targetLayer[0].downsample[1].bias.data = rootLayer[0].downsample[1].bias.data  


def trainclassifier(net, optimizer, criterion, batchSize, dImgTrain, imSize, imageTransform, dClassImgTrain, IdOutput):
    net.train()
    # shuffle images name 
    lImgName = dImgTrain.keys()
    random.shuffle(lImgName)
    # Split the whole list into sublist sizeof batch_size
    for subListImgName in [lImgName[i:i+batchSize] for i in range(0, len(lImgName), batchSize)][:-1]:
        # transform images into tensor
        inputs = torch.Tensor(batchSize, 3, imSize, imSize).cuda()
        for i, imgName in enumerate(subListImgName): inputs[i] = imageTransform(dImgTrain[imgName])
        inputs = Variable(inputs)  
        # list class of the sublist images
        lab = Variable(torch.LongTensor([dClassImgTrain[imgName] for imgName in subListImgName]).cuda())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)[IdOutput]
        loss = criterion(outputs, lab)
        loss.backward()
        optimizer.step()


def testClassifier(net, dImgTest, imSize, imageTransform, dClassImgTrain, IdOutput):
    net.eval()
    nbCorrect = 0
    for imgName in dImgTest:
        inp = torch.Tensor(1,3, imSize, imSize).cuda()
        inp[0] = imageTransform(dImgTest[imgName])

        outputs = net(Variable(inp))[IdOutput]
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.tolist()        
        nbCorrect+= (predicted[0][0] == dClassImgTrain[imgName])

    print "test : #Correct "+str(nbCorrect)+" on "+str(len(dImgTest))+" ("+str(round(float(nbCorrect)/float(len(dImgTest))*100, 1))+"%)"
