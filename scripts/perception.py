#! /usr/bin/env python

import tf
import os
import cv2
import sys
import math
import rospy
import rospkg
import tf2_ros
import numpy as np
import tf2_geometry_msgs
from PIL import Image as img
from image_geometry import PinholeCameraModel
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, PointStamped
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from tf.transformations import quaternion_from_euler, euler_from_quaternion

#Defining the class
class Perception:

	'''
		Initialization Function

		Input -
			1. image_topic - topicname for RGB image
			2. depth_topic - topicname for raw depth image
			3. camera_frame - opticial frame of the camera used
			4. camera_topic - topicname that publishes camera details 
 	'''
    #Init function
	def __init__(self, image_topic = "/kinect/color/image_raw", depth_topic = "/kinect/depth/image_raw", 
				camera_frame = "kinect_optical_frame",	camera_topic = '/kinect/color/camera_info'):

		self.image = None

		#Loading the labels
		rospack = rospkg.RosPack()
		package_path = rospack.get_path('YOLO_3D')
		labelsPath = os.path.sep.join([package_path, 'yolo', 'obj.names'])
		self.pcd_path = os.path.sep.join([package_path, 'meshes'])
		self.LABELS = open(labelsPath).read().strip().split("\n")

		#Derive the paths to the YOLO weights and model configuration
		weightsPath = os.path.sep.join([package_path, 'yolo', "yolov3_best.weights"])
		configPath = os.path.sep.join([package_path, 'yolo', "yolov3-tiny.cfg"])

		# Load our YOLO object detector trained on custom data
		print("[INFO] loading YOLO from disk...")
		self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


		# Transformations
		self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0))
		self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

		#Defining the topics
		self.image_topic = image_topic
		self.depth_topic = depth_topic
		self.camera_frame = camera_frame
		self.camera_topic = camera_topic

		#Defining the subscriber
		rospy.Subscriber(self.image_topic, Image, self.image_cb)

		#Defining the bridge
		self.bridge = CvBridge()

		#Getting the camera info
		self.camera_info = rospy.wait_for_message(self.camera_topic, CameraInfo)

		#Making Camera Model
		self.cam_model = PinholeCameraModel()
		self.cam_model.fromCameraInfo(self.camera_info)


	'''
		Callback function to convert Sensor_msg/Image to CV2 Image using cv2_bridge
		
		Input - 
			1. image - Sensor_msg Image
	'''
	#Image Callback Function
	def image_cb(self, image):
		try:
			#Converting Sensor Image to cv2 Image
			self.image = self.bridge.imgmsg_to_cv2(image, 'bgr8')
		
		#Handling exceptions
		except CvBridgeError as e:
			print(e)


	'''
		Function to convert 2D image coordinates to 3D world coordinates using PinHoleCamera Model and Depth Image
		
		Input - 
			1. cx - X coordinate in image frame
			2. cy - Y coordinte in image frame
	'''
	#Function for converting coordinates to base link frame
	def uv_to_xyz(self, cx, cy):
		
		#Converting to XYZ coordinates
		(x, y, z) = self.cam_model.projectPixelTo3dRay((cx, cy))
		#Normalising
		x, y, z = x/z, y/z, z/z

		#Getting the depth at given coordinates
		depth = rospy.wait_for_message(self.depth_topic, Image)
		depth_img = img.frombytes("F", (depth.width, depth.height), depth.data)
		lookup = depth_img.load()
		d = lookup[cx, cy]

		#Modifying the coordinates
		x, y, z = x*d, y*d, z*d

		#Making Point Stamp Message
		point_wrt_camera = PointStamped()
		point_wrt_camera.header.frame_id = self.camera_frame
		point_wrt_camera.point.x =  x
		point_wrt_camera.point.y =  y
		point_wrt_camera.point.z =  z

		#Transforming
		target_frame = "world"
		source_frame = self.camera_frame
		transform = self.tf_buffer.lookup_transform(target_frame,
											source_frame, #source frame
											rospy.Time(0), #get the tf at first available time
											rospy.Duration(1.0)) #wait for 1 second

		self.point_wrt_world = tf2_geometry_msgs.do_transform_point(point_wrt_camera, transform)
		
	
	'''
		Function to run prediction on input image and compute the cooresponding 3D coordinates of the centroid of detected bounding box

		Input - 
			1. req_label - Class which needs to be identified
		
		Output -
			1. Point stampped message if label is detected else None
	'''
	#Function to predict and return the pose message	
	def predict(self, req_label):

		while(self.image is None): 
			rospy.sleep(1)
		#Getting the image dimensions
		(self.H, self.W) = self.image.shape[:2]

		self.center = (319.5, 239.5)#(self.W/2, self.H/2)
		self.focal_length = 525.0 #self.center[0] / np.tan(60/2 * np.pi / 180)
		self.camera_matrix = np.array([[self.focal_length, 0, self.center[0]],
											[0, self.focal_length, self.center[1]],
											[0, 0, 1]], dtype = "double")

		# determine only the *output* layer names that we need from YOLO
		ln = self.net.getLayerNames()
		ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		#Creating Blob from images
		blob = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
		
		#Feeding the blob as input
		self.net.setInput(blob)
		#Forward pass
		layerOutputs = self.net.forward(ln)

		#Initializing lists of Detected Bounding Boxes, Confidences, and Class IDs
		boxes = []
		confidences = []
		classIDs = []

		#Looping over each of the layer outputs
		for output in layerOutputs:
		    #Looping over each of the detections
		    for detection in output:
		        #Extracting the class ID and confidence
		        scores = detection[5:]
		        classID = np.argmax(scores)
		        confidence = scores[classID]

		        #Filtering out weak predictions
		        if confidence > 0.5:
		            #Scale the Bounding Box Coordinates
		            box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
		            (centerX, centerY, width, height) = box.astype("int")
		            # use the center (x, y)-coordinates to derive the top and
		            # and left corner of the bounding box
		            x = int(centerX - (width / 2))
		            y = int(centerY - (height / 2))
		            #Updating the Lists
		            boxes.append([x, y, int(width), int(height)])
		            confidences.append(float(confidence))
		            classIDs.append(classID)

		#Applying non-max Suppression
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

		#Ensuring at least one detection exists
		if len(idxs) > 0:
			#Looping over the indexes
			print(len(idxs))
			for i in idxs.flatten():
				#Extracting the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])
				#Getting the labels
				label = self.LABELS[classIDs[i]]
				
				#Checking is required label is being detected
				if req_label == label:
					#center coordinates
					cx = x + (w/2)
					cy = y + (h/2)

					#Converting the center cooridnates to base link frame
					self.uv_to_xyz(cx, cy)

					return self.point_wrt_world
					
		
			#Label not found
			return None
