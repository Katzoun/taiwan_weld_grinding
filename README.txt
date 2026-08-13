Point_Cloud_Preprocessing: Surface reconstruction, resampling, and other pre-processing of point clouds scanned by the depth camera

Weld_Recognition&Grinding_Path_Planning(PointCNN): Weld bead recognition and grinding path planning using PointCNN (depth camera)

Quaternion-Eular_Angle_Transform: Converts between rotation matrices / Euler angles and quaternions

Eye-to-Hand_Calibration(SVD+ICP): Point-cloud-alignment-based eye-to-hand calibration method for producing the coordinate-frame transformation matrix

---------------------------------------------------------------------------------------------------------------------------------

PointCNN_Training: PointCNN training main program (does not include the model architecture definition; requires importing the functions from the PointCNN_Training_Functions folder to train). Currently broken -- it imports a pointcnn_seg_1024 module that is not present anywhere in this repo.

---------------------------------------------------------------------------------------------------------------------------------

ABB_Socket_Connection: TCP/IP + Ethernet program for controlling linear motion of an ABB robot arm from a PC. Requires importing the socket_conn_new file from the ABB_6-Axis_Robot folder into the robot's teach pendant to use together with this program.
