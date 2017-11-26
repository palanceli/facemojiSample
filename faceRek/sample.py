# -*- coding:utf-8 -*-

import logging
import os
import unittest
import numpy
import dlib
import cv2
from skimage import io

class MainApp(object):
    def __init__(self):
        # 训练好的人脸关键点检测器数据
        self.predictorPath = 'extdata/shape_predictor_68_face_landmarks.dat'
        # 训练好的ResNet人脸识别模型
        self.faceRecModelPath = 'extdata/dlib_face_recognition_resnet_model_v1.dat'
        self.shapePredictor = dlib.shape_predictor(self.predictorPath) # 人脸关键点监测器
        self.faceRec = dlib.face_recognition_model_v1(self.faceRecModelPath) # 人脸识别模型

        # 组织样本数据 和 待识别数据
        nSamples = 7
        # 样本数据n-0.jpg
        self.candImageFiles = ['images/%d-0.jpg'% (i + 1) for i in range(nSamples) ]
        self.testImageFiles = [] # 待识别数据n-[1, 10].jpg
        for nCand in range(nSamples):
            for idx in range(10):
                testImage = 'images/%d-%d.jpg' % (nCand + 1, (idx + 1))
                if not os.path.exists(testImage):
                    break
                self.testImageFiles.append(testImage)

    def getFaceDescriptor(self, imgFile):
        ''' 提取人脸特征描述项'''
        img = io.imread(imgFile) # 输入单人头像
        # 正脸检测器
        detector = dlib.get_frontal_face_detector()
        # 检测出人脸区域
        faces = detector(img, 1)
        if len(faces) != 1:
            raise Exception('请确保图片是单人的')
        faceRect = faces[0]

        # 关键点检测
        shape = self.shapePredictor(img, faceRect)
        # 描述项提取， 128D向量
        faceDescriptor = self.faceRec.compute_face_descriptor(img, shape)
        # 转换为numpu array
        return numpy.array(faceDescriptor)

    def MainProc(self):
        detector = dlib.get_frontal_face_detector()		# 正脸检测器
        sp = dlib.shape_predictor(self.predictorPath)	# 人脸关键点检测器
        facerec = dlib.face_recognition_model_v1(self.faceRecModelPath) # 人脸识别模型

        candDescriptors = [] # 候选人描述项
        for imgFile in self.candImageFiles: # 提取候选人特征
            logging.debug("Processing {} ...".format(imgFile))
            candDescriptors.append(self.getFaceDescriptor(imgFile))

        for imgFile in self.testImageFiles:
            testDescriptor = self.getFaceDescriptor(imgFile) # 提取待识别人特征

            distList = []   # 待识别人和每位候选的特征差
            for candDesc in candDescriptors:
                distance = numpy.linalg.norm(candDesc - testDescriptor)
                distList.append(distance)

            # 组织成数据结构 {候选人:特征差}
            candDistDict = dict(zip(self.candImageFiles, distList)) 
            # 根据特征差排序
            sortedCandDistDict = sorted(candDistDict.iteritems(), key = lambda d:d[1])
            # 距离待识别人最近的候选
            logging.debug('识别出：%s => %s' % (imgFile, sortedCandDistDict[0][0]))
  
class DLibUT(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

        # 训练好的人脸关键点检测器数据
        self.predictorPath = 'extdata/shape_predictor_68_face_landmarks.dat'
        # 训练好的ResNet人脸识别模型
        self.faceRecModelPath = 'extdata/dlib_face_recognition_resnet_model_v1.dat'

        self.shapePredictor = dlib.shape_predictor(self.predictorPath) # 人脸关键点监测器
        self.faceRec = dlib.face_recognition_model_v1(self.faceRecModelPath) # 人脸识别模型

    def waitToClose(self, img):
        while True:
            cv2.imshow('image', img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        
        cv2.destroyAllWindows()

    def test00(self):
        ''' 跑主流程 '''
        app = MainApp()
        app.MainProc()

    def test01(self):
        ''' 从一张多人图片中标出人脸区域 '''
        img = cv2.imread('images/girls.jpg')
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = dlib.get_frontal_face_detector() # 正脸检测器
        faces = detector(rgbImg, 1) # 返回脸的信息
        for face in faces:  # 框出每张脸
            cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 1)
            logging.debug(face)

        self.waitToClose(img)
      
    def test02(self):
        ''' 打印提取到的人脸描述项  '''
        app = MainApp()
        faceDescriptor = app.getFaceDescriptor('images/1-0.jpg')
        logging.debug(faceDescriptor)

    def test03(self):
        app = MainApp()
        candImageFiles = ['images/%d-0.jpg' % (i + 1) for i in range(2)]
        testImageFile = 'images/1-1.jpg'
        testDesc = app.getFaceDescriptor(testImageFile) # 待识别人描述项
        distList = []

        candDescriptors = [] # 候选人描述项
        for imgFile in candImageFiles: # 提取候选人特征
            logging.debug("Processing {} ...".format(imgFile))
            candDesc = app.getFaceDescriptor(imgFile)
            candDescriptors.append(candDesc)

            distance = candDesc - testDesc
            logging.debug('candDesc - testDesc = %s\n norm=%s' % 
                            (distance, numpy.linalg.norm(distance)))

            distList.append(numpy.linalg.norm(candDesc - testDesc))
        
        candDistDict = dict(zip(candImageFiles, distList))
        logging.debug(candDistDict)
        sortedCandDistDict = sorted(candDistDict.iteritems(), key = lambda d:d[1])
        logging.debug(sortedCandDistDict)
        logging.debug('识别出：%s => %s' % (testImageFile, sortedCandDistDict[0][0]))

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main
    # cmd: python -m unittest sample.DLibUT.test01