# 概要
YOLOで物体検出を行い、それをカプセルネットワークで分類するパッケージです。

# 実行方法
    $ roslaunch realsense_camera r200_nodelet_rgbd.launch  
    $ roslaunch darknet_ros darknet_ros_gdb.launch  
    $ rosrun e_object_recognizer object_recognizer  
realsense_camera、darknet_rosは別パッケージ  

# 入出力
1.物体認識  
入力 /object/recog_req std_msgs/String "オブジェクト名"  
出力 /object/recog_res std_msgs/Bool  
指定したオブジェクトが視界の中に存在するかをBoolianで返す。  

2.物体画像領域検出  
入力  /object/grasp_req std_msgs/String "オブジェクト名"  
出力1 /object/grasp_res std_msgs/Bool  
出力2 /object/image_range e_object_recognizer/ImageRange  
指定したオブジェクトを検出できたどうかを出力1で返し、検出した領域を出力2で返す。  

3.物体検出結果画像生成  
入力 /object/image_generate_req std_msgs/Bool  
出力 /object/image_generate_res std_msgs/Bool  
物体検出結果画像を/images/object_detection_result.pngに保存した後、Trueを返す。  

# その他
weightの一部ファイルが大きくて上がらなかったので、足りない分は研究室まで取りに来るようにお願いします。