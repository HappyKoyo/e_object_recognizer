#!/usr/bin/env python
# -*- coding: utf-8 -*
import rospy
import cv2
import PIL
import numpy as np
import tensorflow as tf
#ros msgs
from std_msgs.msg import Bool,String,Int8
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from darknet_ros_msgs.msg import BoundingBoxes
from e_object_recognizer.msg import ImageRange
#my scripts
from capsNet import CapsNet

class EObjectRecognizer:
    def __init__(self):
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw',Image,self.ImageCB)
        self.bbox_sub  = rospy.Subscriber('/darknet_ros/bounding_boxes',BoundingBoxes,self.BoundingBoxCB)
        self.recog_sub = rospy.Subscriber('/object/recog_req',String,self.RecogReqCB)
        self.grasp_sub = rospy.Subscriber('/object/grasp_req',String,self.GraspReqCB)        
        self.recog_pub = rospy.Publisher('/object/recog_res',Bool,queue_size=1)
        self.grasp_pub = rospy.Publisher('/object/grasp_res',Bool,queue_size=1)
        self.image_range_pub = rospy.Publisher('/object/image_range',ImageRange,queue_size=1)
        
        self.OBJ_RECOG_ROOT = '/home/nvidia/catkin_ws/src/e_object_recognizer'
        self.obj_list = ['attack','beads','bikkle','chipstar','cocacola','cupnoodle','jagariko','pringles','redbull','sevenup']
        self.bridge = CvBridge()
        self.bbox = 'none'

    def ImageCB(self,img):
        self.full_image = img

    def BoundingBoxCB(self,bb):
        self.bbox = bb.boundingBoxes

    def RecogReqCB(self,target):
        full_image = self.bridge.imgmsg_to_cv2(self.full_image,"bgr8")
        full_image = full_image[::-1,:,::-1].copy()
        bb = self.bbox
        for i in range(len(bb)):
            image = full_image[bb[i].ymin:bb[i].ymax,bb[i].xmin:bb[i].xmax]
            image = PIL.Image.fromarray(image)           #pil
            image = image.resize((28,28))                #pil 28*28
            image = np.asarray(image)                    #python array
            image = image.astype(np.float32)/255         #float array
            if target.data == self.obj_list[self.inference(image)]:
                print "Object exists."                               
                self.recog_pub.publish(True)
                return
        self.recog_pub.publish(False)
        print 'Object does not exist.'
        
    def GraspReqCB(self,target):
        full_image = self.bridge.imgmsg_to_cv2(self.full_image,"bgr8")
        full_image = full_image[::-1,:,::-1].copy()
        bb = self.bbox
        for i in range(len(bb)):
            image = full_image[bb[i].ymin:bb[i].ymax,bb[i].xmin:bb[i].xmax]
            image = PIL.Image.fromarray(image)           #pil
            image = image.resize((28,28))                #pil 28*28
            image = np.asarray(image)                    #python array
            image = image.astype(np.float32)/255         #float array
            if target.data == self.obj_list[self.inference(image)]:
                print "Object exists."                               
                obj_image_range = ImageRange()
                obj_image_range.top    = bb[i].ymin
                obj_image_range.bottom = bb[i].ymax
                obj_image_range.left   = bb[i].xmin
                obj_image_range.right  = bb[i].xmax
                self.image_range_pub.publish(obj_image_range)
                return
        self.grasp_pub.publish(False)
        print 'Object does not exist.'
        
    def inference(self,image):
        print "Loading Network.."
        pred_result = []
        image_placeholder = tf.placeholder(tf.float32,
                                           shape=[28,28,3],
                                           name='input_image')
        model = CapsNet()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            weights_dir = self.OBJ_RECOG_ROOT+'/weights'
            supervisor = tf.train.Supervisor(graph=model.graph,logdir=weights_dir,save_model_secs=0)
            with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                supervisor.saver.restore(sess, tf.train.latest_checkpoint(weights_dir))
                pred = model.inference()
                logits_value = sess.run([pred],
                                        feed_dict={
                                            model.image_placeholder:image
                                        })
        pred_result = logits_value[0][0]
        result = np.argmax(pred_result, axis=0, out = None)
        obj_num = result[0][0]
        print 'object is ',self.obj_list[obj_num]
        return obj_num
    
    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('e_object_recognizer')
    obj_recog = EObjectRecognizer()
    obj_recog.main()
