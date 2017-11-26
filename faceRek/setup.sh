currDir=$(cd "$(dirname "$0")"; pwd)
# curl： -C断点续传; -O使用URL中默认的文件名

# 训练好的人脸关键点检测器
curl -C - -O http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# 训练好的ResNet人脸识别模型
curl -C - -O http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

# bzip -k不要删除源文件 -d解压
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d dlib_face_recognition_resnet_model_v1.dat.bz2

rm -f extdata
mkdir extdata
mv shape_predictor_68_face_landmarks.dat extdata/
mv dlib_face_recognition_resnet_model_v1.dat extdata/
