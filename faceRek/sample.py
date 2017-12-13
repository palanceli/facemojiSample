# -*- coding:utf-8 -*-

import logging
import os
import unittest
import numpy
import dlib
import cv2
# from skimage import io
import faceswap
import time
import timeit

class DLibHelper(object):
    def __init__(self):
        self.mParams = {'predictorPath':'extdata/shape_predictor_68_face_landmarks.dat', 
        'faceRecModelPath':'extdata/dlib_face_recognition_resnet_model_v1.dat'}
        self.ReloadFunctions()

    def ReloadFunctions(self):
        self.funcShapePredictor = dlib.shape_predictor(self.mParams['predictorPath'])
        self.funcFaceRecognize = dlib.face_recognition_model_v1(self.mParams['faceRecModelPath'])
        self.funcDetector = dlib.get_frontal_face_detector()

    def GetFaces(self, img):    # 返回所有脸盘
        return self.funcDetector(img, 0)

    def GetFaceLandmarks(self, img, faceRect):  # 返回指定脸盘的landmarks
        keyPts = self.funcShapePredictor(img, faceRect).parts()
        landmarks = numpy.matrix([[p.x, p.y] for p in keyPts])
        return landmarks


class DLibPerfUT(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

    def waitToClose(self, img):
        while True:
            cv2.imshow('image', img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        
        cv2.destroyAllWindows()

    def test01(self):
        dlibHelper = DLibHelper()
        imgFile = 'images/5-1.jpg'
        img = cv2.imread(imgFile)

        time0 = timeit.default_timer()
        faceRects = dlibHelper.GetFaces(img)
        time1 = timeit.default_timer()

        if len(faceRects) == 0:
            raise Exception('face not found!')
        faceRect = faceRects[0]
        time2 = timeit.default_timer()
        landmarks = dlibHelper.GetFaceLandmarks(img, faceRect)
        time3 = timeit.default_timer()

        logging.debug('getFaces:%5.3f, getFaceLandmarks:%5.3f' % (time1 - time0, time3 - time2))


        pts = numpy.array(landmarks, numpy.int32)
        cv2.polylines(img, pts.reshape(-1, 1, 2), True, (0, 255, 0), 2)

        self.waitToClose(img)

    def test02(self):
        ''' 从摄像头读取图像并显示 '''
        dlibHelper = DLibHelper()
        cap = cv2.VideoCapture(0)
        img = cap.read()[1]
        while True:
            img = cap.read()[1]
            
            t0= timeit.default_timer()
            imgZoomed = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC) # 缩小四分之一
            t1= timeit.default_timer()

            faceRects = dlibHelper.GetFaces(imgZoomed)  # 检测人脸rect
            if len(faceRects) == 0:
                continue

            t2= timeit.default_timer()
            faceRect = faceRects[0]
            faceRect = dlib.rectangle(faceRects[0].left()*4, faceRects[0].top()*4, 
                faceRects[0].right()*4, faceRects[0].bottom()*4)
            cv2.rectangle(img, (faceRect.left(), faceRect.top()), (faceRect.right(), faceRect.bottom()), (0, 255, 0), 1)

            landmarks = dlibHelper.GetFaceLandmarks(img, faceRect)  # 提取landmarks
            t3 = timeit.default_timer()
            logging.debug('GetFaces:%5.3f, GetFaceLandmarks:%5.3f' % (t2-t1, t3-t2))


            pts = numpy.array(landmarks, numpy.int32)
            cv2.polylines(img, pts.reshape(-1, 1, 2), True, (0, 255, 0), 2)

            cv2.imshow('image', img)
            key = cv2.waitKey(1)
            if key == 27:
                break


    def test03(self):
        ''' 从摄像头读取图像并显示 '''
        dlibHelper = DLibHelper()
        cap = cv2.VideoCapture(0)
        img = cap.read()[1]
        tracker = dlib.correlation_tracker()
        faceRect = None
        while True:
            img = cap.read()[1]
            
            # logging.debug(tracker.get_position())
            t0 = timeit.default_timer()

            if faceRect is None:
                faceRects = dlibHelper.GetFaces(img)
                if len(faceRects) == 0:
                    continue
                faceRect = dlib.rectangle(faceRects[0].left(), faceRects[0].top(), faceRects[0].right(), faceRects[0].bottom())
                tracker.start_track(img, faceRect)
            else:
                tracker.update(img)
                rect = tracker.get_position()
                faceRect = dlib.rectangle(int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom()))
            t1 = timeit.default_timer()
            cv2.rectangle(img, (faceRect.left(), faceRect.top()), (faceRect.right(), faceRect.bottom()), (0, 255, 0), 1)

            landmarks = dlibHelper.GetFaceLandmarks(img, faceRect)
            t2 = timeit.default_timer()
            logging.debug('GetFaceRect:%5.3f, GetFaceLandmarks:%5.3f' % (t1-t0, t2-t1))

            pts = numpy.array(landmarks, numpy.int32)
            cv2.polylines(img, pts.reshape(-1, 1, 2), True, (0, 255, 0), 2)

            cv2.imshow('image', img)
            key = cv2.waitKey(1)
            if key == 27:
                break

class FaceUtil(object):
    def __init__(self):
        # 训练好的人脸关键点检测器数据
        self.predictorPath = 'extdata/shape_predictor_68_face_landmarks.dat'
        # 训练好的ResNet人脸识别模型
        self.faceRecModelPath = 'extdata/dlib_face_recognition_resnet_model_v1.dat'
        self.ShapePredictor = dlib.shape_predictor(self.predictorPath) # 人脸关键点监测器
        self.FaceRec = dlib.face_recognition_model_v1(self.faceRecModelPath) # 人脸识别模型
        self.Detector = dlib.get_frontal_face_detector()    # 正脸检测器

class FaceRek(object):
    def __init__(self):
        self.faceUtil = FaceUtil()
        self.shapePredictor = self.faceUtil.ShapePredictor
        self.faceRec = self.faceUtil.FaceRec

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

    def getFaceLandmarks(self, imgFile):
        ''' 提取人脸关键点 '''
        img = io.imread(imgFile) # 输入单人头像
        return self.GetFaceLandmarksFromImg(img)

    def GetFaceLandmarksFromImg(self, img):
        # 正脸检测器
        detector = dlib.get_frontal_face_detector()
        # 检测出人脸区域
        faces = detector(img, 1)
        if len(faces) != 1:
            raise Exception('请确保图片是单人的')
        faceRect = faces[0]

        # 提取关键点
        keyPts = self.shapePredictor(img, faceRect).parts()
        landmarks = numpy.matrix([[p.x, p.y] for p in keyPts])
        return landmarks


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
        faceRek = FaceRek()
        faceDescriptor = faceRek.getFaceDescriptor('images/1-0.jpg')
        logging.debug(faceDescriptor)

    def test03(self):
        ''' 从candImages图片中挑出testImage那个人 '''
        faceRek = FaceRek()
        candImageFiles = ['images/%d-0.jpg' % (i + 1) for i in range(7)]
        
        candDescriptors = [] # 候选人描述项
        for imgFile in candImageFiles: # 提取候选人特征
            logging.debug("Processing {} ...".format(imgFile))
            candDesc = faceRek.getFaceDescriptor(imgFile)
            candDescriptors.append(candDesc)
        
        testImageFiles = ['images/%d-1.jpg' % (i + 1) for i in range(7)]
        for testImageFile in testImageFiles:
            testDesc = faceRek.getFaceDescriptor(testImageFile) # 待识别人描述项
            distList = []
            for candDesc in candDescriptors:
                distance = candDesc - testDesc
                # 打印特征值的差和归一值
                # logging.debug('candDesc - testDesc = %s\n norm=%s' % 
                #                 (distance, numpy.linalg.norm(distance)))

                distList.append(numpy.linalg.norm(candDesc - testDesc))
            
            candDistDict = dict(zip(candImageFiles, distList))
            # logging.debug(candDistDict) # 打印{cand : distance}
            sortedCandDistDict = sorted(candDistDict.iteritems(), key = lambda d:d[1])
            # logging.debug(sortedCandDistDict) # 打印按照distance排序的{cand : distance}
            logging.debug('识别出：%s => %s' % (testImageFile, sortedCandDistDict[0][0]))

    def test04(self):
        ''' 识别出单人图片中人脸的关键点 '''
        faceRec = FaceRek()
        imgFile = 'images/1-0.jpg'
        img = cv2.imread(imgFile)
        landmarks = faceRec.getFaceLandmarks(imgFile)
        logging.debug(landmarks)
        pts = numpy.array([landmarks], numpy.int32)
        cv2.polylines(img, pts.reshape(-1, 1, 2), True, (0, 255, 0), 2)
        self.waitToClose(img)

    def test05(self):
        ''' 识别出多人图片中人脸关键点 '''
        img = cv2.imread('images/girls.jpg')
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detector = dlib.get_frontal_face_detector()
        faces = detector(rgbImg, 1)
        for face in faces:
            # 识别出关键点，keyPts的类型是dlib.points
            keyPts = self.shapePredictor(img, face).parts()
            landmarks = numpy.matrix([[p.x,p.y] for p in keyPts]) 
            
            pts = numpy.array([landmarks], numpy.int32)
            cv2.polylines(img, pts.reshape(-1, 1, 2), True, (0, 255, 0), 2)

        self.waitToClose(img)

    def synthesisFaces(self, faceFile1, faceFile2):
        im1, landmarks1 = faceswap.read_im_and_landmarks(faceFile1)
        im2, landmarks2 = faceswap.read_im_and_landmarks(faceFile2)

        M = faceswap.transformation_from_points(landmarks1[faceswap.ALIGN_POINTS],
                                    landmarks2[faceswap.ALIGN_POINTS])

        mask = faceswap.get_face_mask(im2, landmarks2)
        warped_mask = faceswap.warp_im(mask, M, im1.shape)
        combined_mask = numpy.max([faceswap.get_face_mask(im1, landmarks1), warped_mask],
                                axis=0)

        warped_im2 = faceswap.warp_im(im2, M, im1.shape)
        warped_corrected_im2 = faceswap.correct_colours(im1, warped_im2, landmarks1)

        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        return output_im

    def test06(self):
        ''' 将两张人脸合成一张 '''
        imgFile1 = 'images/me.jpg'
        imgFile2 = 'images/moji8.jpg'
        resultFile = 'temp.jpg'

        imgResult = self.synthesisFaces(imgFile1, imgFile2)
        cv2.imwrite(resultFile, imgResult)

        img1 = cv2.imread(imgFile1)
        img2 = cv2.imread(imgFile2)
        imgResult = cv2.imread(resultFile)

        rows1, cols1, channels1 = img1.shape
        rows2, cols2, channels2 = img2.shape
        rows3, cols3, channels3 = imgResult.shape

        MARGIN = 10
        rows = max(rows1, rows2)
        cols = cols1 + cols2 + cols3 + MARGIN * 3
        imgTotal = numpy.zeros((rows, cols, 3), numpy.uint8)
        imgTotal[: rows1, :cols1] = img1
        imgTotal[: rows2, cols1 + MARGIN : cols1 + MARGIN + cols2] = img2
        imgTotal[: rows3, cols1 + cols2 + MARGIN * 2 : cols1 + cols2 + cols3 + MARGIN * 2] = imgResult
        self.waitToClose(imgTotal)
        os.remove(resultFile)


    def getFaceLandmarksFromImg(self, img):
        # 检测出人脸区域
        faces = self.mDetector(img, 1)
        if len(faces) < 1:
            return None
        faceRect = faces[0]
        # logging.debug(faceRect)
        # ltx, lty = faceRect.left(), faceRect.top()
        # rbx, rby = faceRect.right(), faceRect.bottom()
        # faceImg = img[ltx:rbx, lty:rby]
        # 提取关键点
        keyPts = self.mShapePredictor(img, faceRect).parts()
        landmarks = numpy.matrix([[p.x, p.y] for p in keyPts])
        return landmarks

    def test07(self):
        ''' 从摄像头读取图像并显示 '''
        cap = cv2.VideoCapture(0)
        img = cap.read()[1]
        rows, cols, _ = img.shape
        logging.debug('%d, %d' % (rows, cols))
        return
        while True:
            cameraImg = cap.read()[1]
            landmarks = self.getFaceLandmarksFromImg(cameraImg)
            if landmarks is None:
                continue

            pts = numpy.array(landmarks, numpy.int32)
            cv2.polylines(cameraImg, pts.reshape(-1, 1, 2), True, (0, 255, 0), 2)

            cv2.imshow('image', cameraImg)
            key = cv2.waitKey(1)
            if key == 27:
                break

    def test08(self):
        ''' 从图片中识别人脸landmarks，并标出序号 '''
        imgFile = 'images/7-0.jpg'
        img = cv2.imread(imgFile)
        img = cv2.resize(img, None, fx=2.8, fy=2, interpolation=cv2.INTER_CUBIC)
        landmarks = self.getFaceLandmarksFromImg(img)

        pts = numpy.array(landmarks, numpy.int32)
        cv2.polylines(img, pts.reshape(-1, 1, 2), True, (0, 255, 0), 2)


        font = cv2.FONT_HERSHEY_SIMPLEX 
        

        index = 0
        for pt in landmarks:
            x, y = pt[0, 0], pt[0, 1]
            text = '%d' % index
            cv2.putText(img, text, (x, y), font, 0.5, (255, 0, 0))
            index += 1

        self.waitToClose(img)


    def test09(self):
        ''' ，研究如何优化性能 '''
        imgFile = 'images/me.jpg'
        img = cv2.imread(imgFile)
        # img = cv2.resize(img, None, fx=2.8, fy=2, interpolation=cv2.INTER_CUBIC)
        

        rows, cols = img.shape[0], img.shape[1]
        t0 = timeit.default_timer()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        landmarks = self.getFaceLandmarksFromImg(img)
        t1 = timeit.default_timer()
        logging.debug('(%d, %d) %5.3f' % (rows, cols, t1 - t0))

class DLibTrainingUT(unittest.TestCase):
    def setUp(self):
        logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
        logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)

        self.mDetector = dlib.get_frontal_face_detector()   # 正脸检测器

    def waitToClose(self, img):
        while True:
            cv2.imshow('image', img)
            if cv2.waitKey(20) & 0xFF == 27:
                break
        
        cv2.destroyAllWindows()

    def getFaceLandmarksFromImg(self, img, predictorPath):
        # 检测出人脸区域
        faces = self.mDetector(img, 1)
        if len(faces) < 1:
            return None
        faceRect = faces[0]

        self.mShapePredictor = dlib.shape_predictor(predictorPath) # 人脸关键点监测器
        # 提取关键点
        keyPts = self.mShapePredictor(img, faceRect).parts()
        landmarks = numpy.matrix([[p.x, p.y] for p in keyPts])
        return landmarks

    def test01(self):
        ''' 验证自己训练的模型效果 '''
        imgFile = 'images/2-0.jpg'
        img = cv2.imread(imgFile)
        modelPath = '/Users/palance/Documents/SubVersions/dlib/examples/build/Debug/sp.dat'
        landmarks = self.getFaceLandmarksFromImg(img, modelPath)

        pts = numpy.array(landmarks, numpy.int32)
        cv2.polylines(img, pts.reshape(-1, 1, 2), True, (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX 
        
        index = 0
        for pt in landmarks:
            x, y = pt[0, 0], pt[0, 1]
            text = '%d' % index
            cv2.putText(img, text, (x, y), font, 0.5, (255, 0, 0))
            index += 1

        self.waitToClose(img)

if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main
    # cmd: python -m unittest sample.DLibUT.test01