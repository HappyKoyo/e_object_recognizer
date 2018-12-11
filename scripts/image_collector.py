#!/usr/bin/env python
# -*- coding: utf-8 -*
import rospy
import cv2
#ros msgs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

IMAGE_ROOT="/home/demulab/catkin_ws/src/e_object_recognizer/images/"


class EObjectRecognizer:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/wheelrobot/camera1/image_raw',Image,self.ImageCB)

    def ImageCB(self,img):
        self.full_image = img
    
    def main(self):
        rospy.sleep(1)
        while not rospy.is_shutdown():
            full_image = self.bridge.imgmsg_to_cv2(self.full_image,"bgr8")
            cv2.imwrite(IMAGE_ROOT+str(int(rospy.get_time()*1000))+".png",full_image)
            print "image",str(int(rospy.get_time()*1000)),"is saved"
            rospy.sleep(1)

if __name__ == '__main__':
    rospy.init_node('e_object_recognizer')
    obj_recog = EObjectRecognizer()
    obj_recog.main()
