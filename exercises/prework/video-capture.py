# 从视频读取帧保存为图片
from cv2 import cv2 as cv

# 源文件存储路径
file = "D:/jade/develop/YOLO/yolov5/data/GH010060.mp4"
# 读取mp4文件，文件名前一定要用“/”才可被识别不报错
cap = cv.VideoCapture(file)
# cap = cv.VideoCapture(0)  # 当使用的参数为一个数字0的时候代表从摄像头获取视频。

# 文件名从0开始
c = 0
# 保存图片的路径
path = 'D:/jade/develop/YOLO/yolov5/data/videos/images/'

while (1):
    # get a frame
    ret, frame = cap.read()
    # 判断帧率是否不为空
    if ret:
        # show a frame
        cv.imshow("capture", frame)
        # 存储为图像，要手动创建路径文件，不然不显示
        cv.imwrite(path + str(c) + '.jpg', frame)
        # 拼接路径和图片名（以数字有序命名）
        print(path + str(c) + '.jpg', frame)
        # 计数累加1
        c = c + 1
        # cv2.waitKey()，它的参数表示暂停时间，所以这个值越大，视频播放速度越慢，反之，播放速度越快。
        # ord 是转码，因为 opencv 内核是 C++，用的是 ASCⅡ。
        if cv.waitKey(100) & 0xFF == ord('q'):
            # 退出
            break
# 清内存
cap.release()
# 销毁所有窗口
cv.destroyAllWindows()