#! /usr/bin/env python

from perception import Perception


#Calling the main thread
if __name__ == '__main__':

	#Initializing ROS Node
	rospy.init_node("yolo_3d")

	#Getting the parameters from launch file
	image_topic = rospy.get_param('image_topic')
	camera_topic = rospy.get_param('camera_topic')
	depth_topic = rospy.get_param('depth_topic')
	camera_frame = rospy.get_param('camera_frame')

	#Initializing Perception object
	'''
			1. image_topic - topicname for RGB image
			2. depth_topic - topicname for raw depth image
			3. camera_frame - opticial frame of the camera used
			4. camera_topic - topicname that publishes camera details 
	'''
	self.preception_obj = Perception(image_topic = image_topic, 
									depth_topic = camera_topic, 
									camera_frame = depth_topic, 
									camera_topic = camera_frame)

	#Sending to YOLO and getting output coordinates
	#Returns the PointStamped Messgae which contains the 3D coordinate of object in world frame
	obj_pose = self.preception_obj.predict('apple')
	print(obj_pose)
