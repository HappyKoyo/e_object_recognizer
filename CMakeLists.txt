cmake_minimum_required(VERSION 2.8.3)
project(e_object_recognizer)
find_package(catkin REQUIRED COMPONENTS
  #pcl_ros
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)
add_message_files(
  FILES
  ImageRange.msg
)
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)
catkin_package(
  INCLUDE_DIRS #include
#  LIBRARIES e_object_recognizer
CATKIN_DEPENDS message_runtime roscpp rospy std_msgs geometry_msgs
#CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs
#  DEPENDS system_lib
)
include_directories(
/usr/local/include
  ${catkin_INCLUDE_DIRS}
#include/
)

