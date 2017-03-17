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
    """
    Compute the mean value of each RGB channel of a set of images
    """
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
    """
    Compute the standard deviation value of each RGB channel of a set of images
    """
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
    """
    Copy the feature part parameters of a pretrained AlexNet model
    """
    for i, f in enumerate(net.features):
        if type(f) is torch.nn.modules.conv.Conv2d:
            print "copy", f
            f.weight.data = netBase.features[i].weight.data
            f.bias.data = netBase.features[i].bias.data

def copysClassifierParameterAlexnet(net, netBase):
    """
    Copy the classifier part parameters of a pretrained AlexNet model
    """
    for i, c in enumerate(net.classifier):
        if type(c) is torch.nn.modules.linear.Linear :
            if c.weight.size() == netBase.classifier[i].weight.size():
                print "copy", c
                c.weight.data = netBase.classifier[i].weight.data
                c.bias.data = netBase.classifier[i].bias.data 


def copyFeaturesParametersResnet(net, netBase, nbBlock1, nbBlock2, nbBlock3, nbBlock4, typeBlock="Bottleneck"):
    """
    Copy all parameters of a Resnet model from a pretrained model (except for the last fully connected layer)
    typeBlock == "BasicBlock" for resnet18 and resnet34 or "Bottleneck" for resnet50, resnet101 and resnet152
    resnet18: nbBlock1, nbBlock2, nbBlock3, nbBlock4 = 2, 2, 2, 2
    resnet34: nbBlock1, nbBlock2, nbBlock3, nbBlock4 = 3, 4, 6, 3
    resnet50: nbBlock1, nbBlock2, nbBlock3, nbBlock4 = 3, 4, 6, 3
    resnet101: nbBlock1, nbBlock2, nbBlock3, nbBlock4 = 3, 4, 23, 3
    resnet152: nbBlock1, nbBlock2, nbBlock3, nbBlock4 = 3, 8, 36, 3
    see model/resnet.py for more informations about the model
    """

    if typeBlock not in ["BasicBlock", "Bottleneck"]:
        print 'error in the block name, choose "BasicBlock", "Bottleneck"'
        return

    print "copy net.conv1", net.conv1
    net.conv1.weight.data = netBase.conv1.weight.data
    print "copy net.bn1", net.bn1
    net.bn1.weight.data = netBase.bn1.weight.data
    net.bn1.bias.data = netBase.bn1.bias.data

    lLayer = [("layer1", net.layer1, netBase.layer1, nbBlock1),
              ("layer2", net.layer2, netBase.layer2, nbBlock2),
              ("layer3", net.layer3, netBase.layer3, nbBlock3),
              ("layer4", net.layer4, netBase.layer4, nbBlock4)
             ]

    if typeBlock == "BasicBlock":
        for layerName, targetLayer, rootLayer, nbC in lLayer:
            print "copy", layerName, rootLayer
            for i in range(nbC):
                targetLayer[i].conv1.weight.data = rootLayer[i].conv1.weight.data
                targetLayer[i].bn1.weight.data = rootLayer[i].bn1.weight.data
                targetLayer[i].bn1.bias.data = rootLayer[i].bn1.bias.data
                targetLayer[i].conv2.weight.data = rootLayer[i].conv2.weight.data
                targetLayer[i].bn2.weight.data = rootLayer[i].bn2.weight.data
                targetLayer[i].bn2.bias.data = rootLayer[i].bn2.bias.data
            if targetLayer[0].downsample:
                targetLayer[0].downsample[0].weight.data = rootLayer[0].downsample[0].weight.data
                targetLayer[0].downsample[1].weight.data = rootLayer[0].downsample[1].weight.data
                targetLayer[0].downsample[1].bias.data = rootLayer[0].downsample[1].bias.data 

    elif typeBlock == "Bottleneck":
        for layerName, targetLayer, rootLayer, nbC in lLayer:
            print "copy", layerName, rootLayer
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




def trainclassifier(net, optimizer, criterion, batchSize, dImg, imSize, imageTransform, dClassImg, IdOutput):
    """
    Train a net:
        net: the neural network (for example: net = models.alexnet())
        optimizer: the optimizer (for example: optimizer = optim.SGD(alexTunedClassifier.parameters(), lr=0.001, momentum=0.9)) 
        batchSize: size of the batch, dependinf on the GPU memory available (batchSize = 32)
        dImg: dictionary with as key the image name and for values the image open with PIL
        imSize: image size (imSize=225 for alexnet, resnet, ...)
        imageTransform: composition of transformation that will be done on images, 
            For example:
                imageTrainTransform = transforms.Compose([transforms.Scale(300), 
                                                          transforms.RandomCrop(225), 
                                                          transforms.ToTensor(), 
                                                          transforms.Normalize(mean = mean,
                                                                               std = std),
                                                         ])
            see transforms.Compose in pytorch for more information
        dClassImg: dictionary with as key the image name and for value the corresponding class
        IdOutput: position of the output of the net(inputs) result to used in the criterion
    """
    net.train()
    # shuffle images name 
    lImgName = dImg.keys()
    random.shuffle(lImgName)
    # Split the whole list into sublist sizeof batch_size
    for subListImgName in [lImgName[i:i+batchSize] for i in range(0, len(lImgName), batchSize)][:-1]:
        # transform images into tensor
        inputs = torch.Tensor(batchSize, 3, imSize, imSize).cuda()
        for i, imgName in enumerate(subListImgName): inputs[i] = imageTransform(dImg[imgName])
        inputs = Variable(inputs)  
        # list class of the sublist images
        lab = Variable(torch.LongTensor([dClassImg[imgName] for imgName in subListImgName]).cuda())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)[IdOutput]
        loss = criterion(outputs, lab)
        loss.backward()
        optimizer.step()


def testClassifier(net, dImg, imSize, imageTransform, dClassImg, IdOutput):
    """
    Test a neural network on a test dataset
        net: the neural network (for example: net = models.alexnet())
        dImg: dictionary with as key the image name and for values the image open with PIL
        imSize: image size (imSize=225 for alexnet, resnet, ...)
        imageTransform: composition of transformation that will be done on images, 
            For example:
                imageTestTransform = transforms.Compose([transforms.ToTensor(), 
                                                         transforms.Normalize(mean = mean,
                                                                              std = std),
                                                        ]))
            see transforms.Compose in pytorch for more information
        dClassImg: dictionary with as key the image name and for value the corresponding class
        IdOutput: position of the output of the net(inputs) result to used in the criterion
    """
    net.eval()
    nbCorrect = 0
    for imgName in dImg:
        inp = torch.Tensor(1,3, imSize, imSize).cuda()
        inp[0] = imageTransform(dImg[imgName])

        outputs = net(Variable(inp))[IdOutput]
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.tolist()        
        nbCorrect+= (predicted[0][0] == dClassImg[imgName])

    print "test : #Correct "+str(nbCorrect)+" on "+str(len(dImg))+" ("+str(round(float(nbCorrect)/float(len(dImg))*100, 1))+"%)"

