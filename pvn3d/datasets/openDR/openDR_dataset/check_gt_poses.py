import rospy
import yaml
from geometry_msgs.msg import Pose, Twist, PoseArray, PoseStamped, Point, Quaternion, Vector3
import tf
br = tf.TransformBroadcaster()
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.io as sio

mat = np.zeros((3,4,10,2304))
for i in range(0,2304):
	mat[:,:,:,i] = sio.loadmat('./meta/'+str(i)+'-meta.mat')['poses']

rospy.init_node('gt_pub')


for j in range(0,mat.shape[3]):
	if rospy.is_shutdown():
		break
	r = mat[:3, :3, :, j]
	t = mat[:3, 3, :, j]
	print j
	
	#br.sendTransform(( X, Y, Z), (cam_quat[0], cam_quat[1], cam_quat[2], cam_quat[3]),rospy.Time.now(), 'camera_link',"world")
	
	for i in range(0,10):
		obj_cam_or = R.from_dcm(r[:,:,i])
		o2c_or = obj_cam_or.as_quat()
		br.sendTransform((t[0,i], t[1,i] , t[2,i]),(o2c_or[0], o2c_or[1], o2c_or[2], o2c_or[3]),rospy.Time.now(), 'object'+str(i),"camera_link")
	rospy.sleep(0.05)

