# 从视频读取帧保存为图片
from cv2 import cv2 as cv

cap = cv.VideoCapture("D:/jade/develop/YOLO/yolov5/data/videos/GH010060.mp4")  #读取mp4文件

# cap = cv.VideoCapture(0)  # 当使用的参数为一个数字0的时候代表从摄像头获取视频。

c=0  #文件名从0开始

while(1):
   ret, frame = cap.read()  # get a frame
   cv.imshow("capture", frame)  # show a frame
   cv.imwrite("D:/jade/develop/YOLO/yolov5/runs/images/"+str(c) + '.jpg', frame)  #存储为图像，要手动创建images文件，不然不显示
   c=c+1
   if cv.waitKey(0) & 0xFF == ord('q'):
      break
cap.release()
cv.destroyAllWindows()
