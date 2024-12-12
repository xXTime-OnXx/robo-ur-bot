#! /usr/bin/env python3

import rospy
import numpy as np
import math
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from urdf_parser_py.urdf import URDF



class ForwardKinematics:

    def __init__(self):  
        # we create two publishers in our ROS node
        # one to publish the angles of the joints (joint states)
        # and another for visualizing a target pose (to see whether your calculations are correct)
        self.joint_state_publisher = rospy.Publisher('my_joint_states', JointState, queue_size=10) 
        self.pose_publisher = rospy.Publisher('my_pose', PoseStamped, queue_size=10)
        rospy.init_node('my_joint_state_publisher')
        self.rate = rospy.Rate(10)
        rospy.sleep(3.0)


    def run(self):
        # to get the kinematic chain with the joints and the corresponding parameters, we use the urdf parser
        # documentation of urdf_parser_py see http://wiki.ros.org/urdfdom_py
        robot = URDF.from_parameter_server()
        root = robot.get_root()
        tip = "tool0"
        joint_names = robot.get_chain(root, tip, joints=True, links=False, fixed=False)
        # the properties of a given joint / link can be obtained with the joint_map
        # see http://wiki.ros.org/urdf/XML/joint

        joint_angles = [0, -3.14/2, 0, 0, 0, 0] # in radians

        # create the joint state messages
        js = JointState()
        js.name = joint_names      
        js.position = joint_angles

        end_effector_pose = self.calculate_forward_kinematics(joint_angles, joint_names, robot)
        
        if end_effector_pose:
            target_pose_message = self.get_pose_message_from_matrix(end_effector_pose)
            print(target_pose_message)
        else:
            rospy.logerr("error, no target pose calculated, use identity matrix instead")
            target_pose_message = self.get_pose_message_from_matrix(np.identity(4))
            

        # publish the joint state values and the target pose
        while not rospy.is_shutdown():
            self.joint_state_publisher.publish(js)
            self.pose_publisher.publish(target_pose_message)
            self.rate.sleep()
 
    def calculate_forward_kinematics(self, joint_positions, joint_names, robot):
        TR1 = self.R_z(joint_positions[0])
        TM1 = self.E(z = 0.15185)

        TR2 = self.R_y(joint_positions[1])
        TM2 = self.E(x = -0.24355)
        
        TR3 = self.R_y(joint_positions[2])
        TM3 = self.E(x = -0.2132)
        
        TR4 = self.R_y(joint_positions[3])
        TM4 = self.E(y = -0.13105)
        
        TR5 = self.R_z(joint_positions[4])
        TM5 = self.E(z = -0.08535)
        
        TR6 = self.R_y(joint_positions[5])
        TM6 = self.E(y = -0.0921)
        

        T1 = TR1 @ TM1 
        T2 = T1 @ TR2 @ TM2 
        T3 = T2 @ TR3 @ TM3
        T4 = T3 @ TR4 @ TM4 
        T5 = T4 @ TR5 @ TM5
        TR = T5 @ TR6 @ TM6
        
        print(TR)
        return TR.tolist()        
    
        
    def R_x(self, theta):
        return np.array([
            [1,         0,              0,              0],
            [0,         np.cos(theta),  np.sin(theta),  0],
            [0,         -np.sin(theta), np.cos(theta),  0],
            [0,         0,              0,              1]
        ])
        
    def R_y(self, theta):
        return np.array([
            [np.cos(theta),     0,          -np.sin(theta),     0],
            [0,                 1,          0,                  0],
            [np.sin(theta),     0,          np.cos(theta),      0],
            [0,                 0,          0,                  1]
        ])
        
    def R_z(self, theta):
        return np.array([
            [np.cos(theta),     -np.sin(theta),     0,      0],
            [np.sin(theta),     np.cos(theta),      0,      0],
            [0,                 0,                  1,      0],
            [0,                 0,                  0,      1]
        ])
        
    def E(self, x = 0, y = 0, z = 0):
        return np.array([
            [1,     0,      0,      x],
            [0,     1,      0,      y],
            [0,     0,      1,      z],
            [0,     0,      0,      1]
        ])

    # the following function creates a PoseStamped message from a homogeneous matrix
    def get_pose_message_from_matrix(self, matrix):

        """Return pose msgs from homogeneous matrix
        matrix : homogeneous matrix 4x4
        """
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose = pose_stamped.pose
        pose.position.x = matrix[0][3]
        pose.position.y = matrix[1][3]
        pose.position.z = matrix[2][3]

        q = self.get_quaternion_from_matrix(matrix)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose_stamped

    # the ROS message type PoseStamped uses quaternions for the orientation
    def get_quaternion_from_matrix(self, matrix):
        """Return quaternion from homogeneous matrix
        matrix : homogeneous matrix 4x4
        """
        q = np.empty((4,), dtype=np.float64)
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        t = np.trace(M)
        if t > M[3, 3]:
            q[3] = t
            q[2] = M[1, 0] - M[0, 1]
            q[1] = M[0, 2] - M[2, 0]
            q[0] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
        return q


if __name__ == '__main__':
    fk = ForwardKinematics()
    fk.run()
    
    
