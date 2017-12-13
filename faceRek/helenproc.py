# -*- coding:utf-8 -*-

import logging
import os
import unittest
import numpy
import dlib
import cv2
import faceswap
import xml.sax

class FaceData(object):
    def __init__(self):
        self.mImgName = None
        self.mPartList = []

    def LoadFromTxt(self, txtPath):
        with open(txtPath, 'r') as f:
            nLine = 0
            for line in f:
                line = line.strip('\r\n')
                if nLine == 0:
                    self.mImgName = line
                else:
                    x, y = line.split(' , ')
                    part = {'name':'%03d' % (nLine - 1), 'x':int(float(x)), 'y':int(float(y)) }
                    self.mPartList.append(part)
                nLine += 1

class HelenProc(object):
    def __init__(self):
        pass

    def ConvTxt2Xml(self, txtDir, xmlPath):
        ''' 将txtDir目录下的txt annotation文件转成xml '''
        for root, dirs, files in os.walk(txtDir):
            for name in files:
                if name.endswith('.txt'):
                    pass


class MainUT(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

        # 训练好的人脸关键点检测器数据
        self.mPredictorPath = 'extdata/shape_predictor_68_face_landmarks.dat'
        self.mShapePredictor = dlib.shape_predictor(self.mPredictorPath) # 人脸关键点监测器
        self.mDetector = dlib.get_frontal_face_detector()   # 正脸检测器

    def waitToClose(self, img):
        while True:
            cv2.imshow('image', img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        
        cv2.destroyAllWindows()

    def test01(self):
        ''' 加载一张图，并把标注在对应位置画出来 '''
        txtFile = '/Users/palance/Downloads/FaceDataset/build/annotation/1515.txt'
        faceData = FaceData()
        faceData.LoadFromTxt(txtFile)

        imgFile = '%s/%s.jpg' % ('/Users/palance/Downloads/FaceDataset/build/train', faceData.mImgName)
        if not os.path.exists(imgFile):
            logging.error('file not found:%s' % imgFile)
            return

        landmarks = []
        for part in faceData.mPartList:
            landmarks.append((part['x'], part['y']))

        pts = numpy.array([landmarks], numpy.int32)
        img = cv2.imread(imgFile)
        cv2.polylines(img, pts.reshape(-1, 1, 2), True, (0, 255, 0))

        self.waitToClose(img)


## 处理ibug的xml文件

class ImagesHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.mImages = []
        self.mCurrImage = None
        self.mCurrBox = None
        self.mCurrPart = None

    def startElement(self, tag, attributes):
        if tag == 'image':
            file = attributes['file']
            self.mCurrImage = {'file':file, 'box':[]}
        elif tag == 'box':
            top = int(attributes['top'])
            left = int(attributes['left'])
            width = int(attributes['width'])
            height = int(attributes['height'])
            self.mCurrBox = {'top':top, 'left':left, 'width':width, 'height':height, 'part':[]}
        elif tag == 'part':
            name = int(attributes['name'])
            x = int(attributes['x'])
            y = int(attributes['y'])
            self.mCurrBox['part'].append({'name':name, 'x':x, 'y':y})

    def endElement(self, tag):
        if tag == 'box':
            self.mCurrImage['box'].append(self.mCurrBox)
        elif tag == 'image':
            self.mImages.append(self.mCurrImage)

class IBugUT(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

    def test01(self):
        ''' 验证ImagesHandler的正确性 '''
        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        handler = ImagesHandler()
        parser.setContentHandler(handler)
        parser.parse('/Users/palance/Downloads/FaceDataset/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train2.xml')
        for image in handler.mImages:
            file = image['file']
            logging.debug('file:%s' % file)
            for box in image['box']:
                top, left, width, height = box['top'], box['left'], box['width'], box['height']
                logging.debug('-- %d, %d, %d, %d' % (top, left, width, height))
                i = 0
                msg = ''
                for part in box['part']:
                    x, y = part['x'], part['y']
                    msg += '(%3d, %3d) ' % (x, y)
                    i += 1
                    if i % 5 == 0:
                        logging.debug('---- %s' % msg)
                        msg = ''

    def test02(self):
        ''' 截取前100条数据 '''
        cTrainItems = 10000
        cLandmarks = 68
        srcPath = '/Users/palance/Downloads/FaceDataset/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml'
        desPath = '/Users/palance/Downloads/FaceDataset/result/labels_ibug_%d_%dlm_train.xml' % (cTrainItems, cLandmarks)
        parser = xml.sax.make_parser()
        parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        handler = ImagesHandler()
        parser.setContentHandler(handler)
        parser.parse(srcPath)
        iImage = 0

        with open(desPath, 'wb') as f:
            f.write('<dataset>\n<images>\n')
            for image in handler.mImages:
                file = image['file']
                f.write("  <image file='%s'>\n" % file)
                for box in image['box']:
                    top, left, width, height = box['top'], box['left'], box['width'], box['height']
                    f.write("    <box top='%d' left='%d' width='%d' height='%d'>\n" % (top, left, width, height))
                    iPart = 0
                    for part in box['part']:
                        name, x, y = part['name'], part['x'], part['y']
                        f.write("      <part name='%02d' x='%d' y='%d'/>\n" % (name, x, y))
                        iPart += 1
                        if iPart > cLandmarks:
                            break
                    f.write("    </box>\n")
                f.write("  </image>\n")
                iImage += 1
                if iImage >= cTrainItems:
                    break
            logging.debug('%d images' % iImage)
            f.write('</images>\n</dataset>')



if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main
    # cmd: python -m unittest sample.DLibUT.test01