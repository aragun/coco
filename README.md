# coco
from pycocotools.coco import COCO

import os
from multiprocessing import Pool
from itertools import repeat
import imageio
from shutil import copyfile, rmtree
import time

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error')
            
class COCOExtractor:
    
    coco = None   
    imageFolder = None
    imagePath = None
    maskPath = None

    def createImageMaskPaths(self, dstDir, dataType):
        maskPath = os.path.join(dstDir, '{}_masks'.format(dataType))
        imagePath = os.path.join(dstDir, '{}_images'.format(dataType))

        if os.path.exists(maskPath):
            rmtree(maskPath)
        if os.path.exists(imagePath):
            rmtree(imagePath)

        createFolder(maskPath)
        createFolder(imagePath)  
        return maskPath, imagePath

    def processImg(self, img):
        copyfile(os.path.join(self.imageFolder, img['file_name']), os.path.join(self.imagePath, '{}.jpg'.format(img['id'])))

    def processAnn(self, ann):
        mask = self.coco.annToMask(ann)
        mask[mask == 1] = 255
        imageio.imwrite(os.path.join(self.maskPath, '{}.jpg'.format(ann['image_id'])), mask)

    def listFiles(self, srcDir, logFile):
        with open(logFile, 'w') as writer:
            for root, dirs, files in os.walk(srcDir):
                for file in files:
                    writer.write(file+'\n')



    # called once each for training and validation
    # TODO: separate part that gets anns so that it can be used for other annotations
    def extractCocoImageAndMask(self, srcDir, dataType, cocoCatNms, dstDir, version):
        # instantiate COCO using annFile
        annFile='{}/annotations/instances_{}{}.json'.format(srcDir,dataType,version)
        self.imageFolder='{}/images/{}{}'.format(srcDir, dataType, version)
        print('Using images from {}'.format(self.imageFolder))
        print('Using annotation file {}'.format(annFile))

        self.coco = COCO(annFile)

        # create mask and image folders
        self.maskPath, self.imagePath = self.createImageMaskPaths(dstDir, dataType)

        catIds = self.coco.getCatIds(catNms=cocoCatNms);
        imgIds = self.coco.getImgIds(catIds=catIds);
        imgs = self.coco.loadImgs(imgIds) # imgs is a list of dictionaries
        
        # TODO: count sample size for each category
        print('{} images in {}_{}'.format(len(imgs), dataType, version))
        annIds = self.coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=False)
        anns = self.coco.loadAnns(annIds) # anns is a list of annotations

        Pool(4).map(self.processImg, imgs)
        Pool(4).map(self.processAnn, anns)

        # create text file with list of filenames
        self.listFiles(self.imagePath, '{}/{}_images.txt'.format(dstDir, dataType))
        self.listFiles(self.maskPath, '{}/{}_masks.txt'.format(dstDir, dataType))

    def extractCoco(self, srcDir, cocoCatNms, dstDir, version):
        print('Creating dataset at {}'.format(dstDir))
        #create train_images, train_masks, val_images, val_masks (folder and .txt for each)
        start = time.time()
        self.extractCocoImageAndMask(srcDir, 'val', cocoCatNms, dstDir, version)
        mid = time.time()
        print('Done with extracting validation set in {} seconds\n'.format(mid-start))

        self.extractCocoImageAndMask(srcDir, 'train', cocoCatNms, dstDir, version)
        print('Done with extracting training set in {} seconds\n'.format(time.time()-mid))

dataDir = '/home/cvmldevalpha/Desktop/cocoapi/'
x = COCOExtractor()
start = time.time()
x.extractCoco(dataDir, ['person', 'chair'], '/home/cvmldevalpha/Desktop/segment', '2017')
end = time.time()
print('{} seconds elapsed in total.'.format(end-start))
