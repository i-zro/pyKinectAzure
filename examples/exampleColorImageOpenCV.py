import sys
sys.path.insert(1, '../pyKinectAzure/')

import numpy as np
from pyKinectAzure import pyKinectAzure, _k4a
import cv2

# MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
              "Background": 15}

POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
              ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
              ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

# Path to the module
# TODO: Modify with the path containing the k4a.dll from the Azure Kinect SDK
modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll' 
# under x86_64 linux please use r'/usr/lib/x86_64-linux-gnu/libk4a.so'
# In Jetson please use r'/usr/lib/aarch64-linux-gnu/libk4a.so'

# 각 파일 path
protoFile = "../caffe/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "../caffe/pose_iter_160000.caffemodel"

# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# 위의 path에 있는 network 불러오기
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

if __name__ == "__main__":

	# Initialize the library with the path containing the module
	pyK4A = pyKinectAzure(modulePath)

	# Open device
	pyK4A.device_open()

	# Modify camera configuration
	device_config = pyK4A.config
	device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
	print(device_config)

	# Start cameras using modified configuration
	pyK4A.device_start_cameras(device_config)

	k = 0
	while True:
		# Get capture
		pyK4A.device_get_capture()

		# Get the color image from the capture
		color_image_handle = pyK4A.capture_get_color_image()

		# Check the image has been read correctly
		if color_image_handle:

			# Read and convert the image data to numpy array:
			color_image = pyK4A.image_convert_to_numpy(color_image_handle)
			# 이미지 읽어오기
			image = color_image
			# frame.shape = 불러온 이미지에서 height, width, color 받아옴
			imageHeight, imageWidth, _ = image.shape

			# network에 넣기위해 전처리
			inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False,
											crop=False)

			# network에 넣어주기
			net.setInput(inpBlob)

			# 결과 받아오기
			output = net.forward()

			# output.shape[0] = 이미지 ID, [1] = 출력 맵의 높이, [2] = 너비
			H = output.shape[2]
			W = output.shape[3]

			# 키포인트 검출시 이미지에 그려줌
			points = []
			for i in range(0, 15):
				# 해당 신체부위 신뢰도 얻음.
				probMap = output[0, i, :, :]

				# global 최대값 찾기
				minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

				# 원래 이미지에 맞게 점 위치 변경
				x = (imageWidth * point[0]) / W
				y = (imageHeight * point[1]) / H

				# 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로
				if prob > 0.1:
					cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=10,
							   lineType=cv2.FILLED)  # circle(그릴곳, 원의 중심, 반지름, 색)
					cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
								lineType=cv2.LINE_AA)
					points.append((int(x), int(y)))
				else:
					points.append(None)

			# Plot the image
			cv2.namedWindow('Color Image',cv2.WINDOW_NORMAL)
			cv2.imshow("Color Image",image)





			k = cv2.waitKey(1)

			# Release the image
			pyK4A.image_release(color_image_handle)

		pyK4A.capture_release()

		if k==27:    # Esc key to stop
			break

	pyK4A.device_stop_cameras()
	pyK4A.device_close()


