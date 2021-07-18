import sys
sys.path.insert(1, '../pyKinectAzure/')

import numpy as np
from pyKinectAzure import pyKinectAzure, _k4a
import cv2

# SDK의 k4a.dll 파일 있는 Path 찾아서 바꿔줄 것!
modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll'

if __name__ == "__main__":

	# 모듈 패스를 통해 SDK 열어주기
	pyK4A = pyKinectAzure(modulePath)
	# device 열기
	pyK4A.device_open()
	# configuration 설정
	device_config = pyK4A.config
	device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
	device_config.depth_mode = _k4a.K4A_DEPTH_MODE_WFOV_2X2BINNED
	print(device_config)

	# configuration에 담긴 내용을 바탕으로 카메라 켜기
	pyK4A.device_start_cameras(device_config)
	# recording 하기
	pyK4A.start_recording("test.mkv")

	k = 0
	while True:
		pyK4A.update()

		depth_image_handle = pyK4A.capture_get_depth_image()

		if depth_image_handle:

			depth_image = pyK4A.image_convert_to_numpy(depth_image_handle)
			depth_color_image = cv2.convertScaleAbs (depth_image, alpha=0.05)  #alpha is fitted by visual comparison with Azure k4aviewer results  
			depth_color_image = cv2.applyColorMap(depth_color_image, cv2.COLORMAP_JET)
			cv2.namedWindow('Colorized Depth Image',cv2.WINDOW_NORMAL)
			cv2.imshow('Colorized Depth Image',depth_color_image)
			k = cv2.waitKey(1)

			pyK4A.image_release(depth_image_handle)

		pyK4A.capture_release()

		if k==27:    # Esc 눌러서 녹화 종료
			break

	pyK4A.device_stop_cameras()
	pyK4A.device_close()
	pyK4A.stop_recording()
