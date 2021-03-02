from PyQt5 import QtWidgets,QtCore,QtGui
import pyqtgraph as pg
import os
from sys import platform
import sys
import argparse
import cv2
import traceback

class MainUi(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainUi,self).__init__(parent)
        self.setWindowTitle("关键点坐标")
        self.main_widget = QtWidgets.QWidget()  # 创建一个主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建一个网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置主部件的布局为网格
        self.setCentralWidget(self.main_widget)  # 设置窗口默认部件

        self.plot_widget = QtWidgets.QWidget()  # 实例化一个widget部件作为K线图部件
        self.plot_layout = QtWidgets.QGridLayout()  # 实例化一个网格布局层
        self.plot_widget.setLayout(self.plot_layout)  # 设置线图部件的布局层
        self.plot_plt = pg.PlotWidget()  # 实例化一个绘图部件
        self.plot_plt.showGrid(x=True, y=True)  # 显示图形网格
        self.plot_layout.addWidget(self.plot_plt)  # 添加绘图部件到线图部件的网格布局层
        # 将上述部件添加到布局层中
        self.main_layout.addWidget(self.plot_widget, 1, 0, 3, 3)

        self.setCentralWidget(self.main_widget)
        self.data_list1 =[]
        self.data_list2 = []
        self.data_list3 = []
        self.data_list4 = []
        self.data_list5 = []
        self.data_list6 = []


    def openpose_detect(self):
        cap = cv2.VideoCapture("E:/joint_angle_measurement/video/跑步.MP4")
        dir_path = "E:/openpose-1.5.0/build/python/openpose"
        if not os.path.exists("E:\\joint_angle_measurement\\input\\frontraise"):
            os.makedirs("E:\\joint_angle_measurement\\input\\frontraise")
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release');
                os.environ['PATH'] = os.environ[
                                         'PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../python');
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e
        # Flags调节openpose的参数
        parser = argparse.ArgumentParser()
        # parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = dict()
        params["model_folder"] = "E:/openpose-1.5.0/models/"
        params["net_resolution"] = "-1x320"
        params["number_people_max"] = 1
        # params["video"]="E:/joint_angle_measurement/video/libin2.mp4"
        # params["part_to_show"] = 0

        params["render_threshold"] = 0.6
        params["render_pose"] = 2
        params["display"] = -1
        params["model_pose"] = "BODY_25"
        # params["frame_step"] = 5
        params["body"] = 1
        params["hand"] = False
        params["write_json"] = ""

        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1]) - 1:
                next_item = args[1][i + 1]
            else:
                next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-', '')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-', '')
                if key not in params: params[key] = next_item
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        datum = op.Datum()

        while 1:
            ret, frame = cap.read()
            try:
                datum.cvInputData = frame
                opWrapper.emplaceAndPop([datum])
                if datum.poseKeypoints.shape == (1, 25, 3):
                    posekeypoints = datum.poseKeypoints[0]
                else:
                    posekeypoints = {}
                lbigtoe_x = posekeypoints[19,0]
                lbigtoe_y = posekeypoints[19,1]
                lsmalltoe_x = posekeypoints[20,0]
                lsmalltoe_y = posekeypoints[20,1]
                lheel_x = posekeypoints[21,0]
                lheel_y=posekeypoints[21,1]
                rbigtoe_x = posekeypoints[22, 0]
                rbigtoe_y = posekeypoints[22, 1]
                rsmalltoe_x = posekeypoints[23, 0]
                rsmalltoe_y = posekeypoints[23, 1]
                rheel_x = posekeypoints[24, 0]
                rheel_y = posekeypoints[24, 0]
                self.data_list1.append(lbigtoe_y)
                self.data_list2.append(lsmalltoe_y)
                self.data_list3.append(lheel_y)
                self.data_list4.append(rbigtoe_y)
                self.data_list5.append(rsmalltoe_y)
                self.data_list6.append(rheel_y)

                print(self.data_list1)
                self.plot_plt.plot().setData(self.data_list1, pen='g',name="lbigtoe")
                self.plot_plt.plot().setData(self.data_list2, pen='y',name="lsmalltoe")
                self.plot_plt.plot().setData(self.data_list3, pen='r',name='lheel')
                cv2.namedWindow("frame", 0)
                cv2.resizeWindow("frame", 1600, 900)  # 设置长和宽
                cv2.imshow('frame', datum.cvOutputData)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
            except Exception as e:
                print(e)
if __name__=='__main__':
    app = QtWidgets.QApplication(sys.argv)
    form =MainUi()
    form.show()
    form.openpose_detect()
    sys.exit(app.exec_())