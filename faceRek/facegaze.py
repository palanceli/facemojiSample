# -*- coding:utf-8 -*-

import logging
import os
import unittest
import numpy
import dlib
import cv2


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

    def getFaceLandmarksFromImg(self, img):
        # 检测出人脸区域
        faces = self.mDetector(img, 1)
        if len(faces) < 1:
            return None
        faceRect = faces[0]

        # 提取关键点
        keyPts = self.mShapePredictor(img, faceRect).parts()
        landmarks = numpy.matrix([[p.x, p.y] for p in keyPts])
        return landmarks

    def test01(self):
        ''' 从摄像头读取图像并显示 '''
        cap = cv2.VideoCapture(0)
        # Anthropometric for male adult
        # Relative position of various facial feature relative to sellion
        # Values taken from https://en.wikipedia.org/wiki/Human_head
        # X points forward
        P3D_SELLION = (0., 0.,0.);
        P3D_RIGHT_EYE = (-20., -65.5,-5.);
        P3D_LEFT_EYE = (-20., 65.5,-5.);
        P3D_RIGHT_EAR = (-100., -77.5,-6.);
        P3D_LEFT_EAR = (-100., 77.5,-6.);
        P3D_NOSE = (21.0, 0., -48.0);
        P3D_STOMMION = (10.0, 0., -75.0);
        P3D_MENTON = (0., 0.,-133.0);

        objp = numpy.array([ P3D_SELLION,
            P3D_RIGHT_EYE, P3D_LEFT_EYE,
            P3D_RIGHT_EAR, P3D_LEFT_EAR,
            P3D_MENTON, P3D_NOSE, P3D_STOMMION,
            ], numpy.float64)

        # FACIAL_FEATURE_INDEX
        NOSE=30,
        RIGHT_EYE=36,
        LEFT_EYE=45,
        RIGHT_SIDE=0,
        LEFT_SIDE=16,
        EYEBROW_RIGHT=21,
        EYEBROW_LEFT=22,
        MOUTH_UP=51,
        MOUTH_DOWN=57,
        MOUTH_RIGHT=48,
        MOUTH_LEFT=54,
        SELLION=27,
        MOUTH_CENTER_TOP=62,
        MOUTH_CENTER_BOTTOM=66,
        MENTON=8

        cameraMtx = numpy.array([455., 0.0, -1.0, 0.0, 455., -1.0, 0.0, 0.0, 1.0], numpy.float32)

        while True:
            cameraImg = cap.read()[1]
            landmarks = self.getFaceLandmarksFromImg(cameraImg)
            if landmarks is None:
                continue

            if cameraMtx[2] < 0:
                rows, cols, channels = cameraImg.shape
                cameraMtx[2] = cols / 2
                cameraMtx[5] = rows / 2

            stomion = (landmarks[MOUTH_CENTER_TOP] + landmarks[MOUTH_CENTER_BOTTOM]) * 0.5
            pts = numpy.array(landmarks, numpy.int32)
            cv2.polylines(cameraImg, pts.reshape(-1, 1, 2), True, (0, 255, 0), 2)

            
            detectedPoints = numpy.array([landmarks[SELLION],
                landmarks[RIGHT_EYE], landmarks[LEFT_EYE],
                landmarks[RIGHT_SIDE], landmarks[LEFT_SIDE],
                landmarks[MENTON], landmarks[NOSE], stomion,
                ], numpy.float64)

            _, rvec, tvec, _ = cv2.solvePnPRansac(objp, detectedPoints, cameraMtx.reshape(3, 3), None)
            logging.debug('(%5.2f, %5.2f, %5.2f),  (%5.2f, %5.2f, %5.2f)' % (rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]))
            cv2.imshow('image', cameraImg)
            key = cv2.waitKey(1)
            if key == 27:
                break


if __name__ == '__main__':
    logFmt = '%(asctime)s %(lineno)04d %(levelname)-8s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=logFmt, datefmt='%H:%M',)
    unittest.main
    # cmd: python -m unittest sample.DLibUT.test01