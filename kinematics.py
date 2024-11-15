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

    def dh_transform(self, a, alpha, d, theta):
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        ct = np.cos(theta)
        st = np.sin(theta)

        res = np.array([[ct, -st, 0, a],
                        [st*ca, ct*ca, -sa, -d*sa],
                        [st*sa, ct*sa, ca, d*ca],
                        [0, 0, 0, 1]])
        return res


    def run(self):
        # to get the kinematic chain with the joints and the corresponding parameters, we use the urdf parser
        # documentation of urdf_parser_py see http://wiki.ros.org/urdfdom_py
        robot = URDF.from_parameter_server()
        root = robot.get_root()
        tip = "tool0"
        joint_names = robot.get_chain(root, tip, joints=True, links=False, fixed=False)
        # the properties of a given joint / link can be obtained with the joint_map
        # see http://wiki.ros.org/urdf/XML/joint

        joint_angles = [0,-3.14/2,3.14/5,0,0,0] # in radians
   
        # create the joint state messages
        js = JointState()
        js.name = joint_names      
        js.position = joint_angles

        end_effector_pose = self.calculate_forward_kinematics(joint_angles)
        rospy.logerr("end effector pose: {}".format(end_effector_pose))
        
        if end_effector_pose is not None:
            target_pose_message = self.get_pose_message_from_matrix(end_effector_pose)
        else:
            rospy.loginfo("error, no target pose calculated, use identity matrix instead")
            target_pose_message = self.get_pose_message_from_matrix(np.identity(4))

        # publish the joint state values and the target pose
        while not rospy.is_shutdown():
            self.joint_state_publisher.publish(js)
            self.pose_publisher.publish(target_pose_message)
            self.rate.sleep()
 
    def calculate_forward_kinematics(self,joint_angles):
        d_n = [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.1761]
        a_n = [0.0, 0.0, -0.24355, -0.2132, 0, 0]
        alpha_n = [0.0, 3.14/2, 0.0, 0.0, 3.14/2, -3.14/2]
        # to implement, should return a 4x4 homogeneous matrix that corresponds to the pose of the end effector
        res_homogeneous_matrix = np.identity(4)
        # create the homogeneous matrix
        for i in range(6):
            res_homogeneous_matrix = np.dot(res_homogeneous_matrix,self.dh_transform(a_n[i], alpha_n[i], d_n[i], joint_angles[i]))
        return res_homogeneous_matrix
    
    def get_position_and_rotation(self, joint_angles):
        pose = self.calculate_forward_kinematics(joint_angles)
        return pose[:3, 3], pose[:3, :3]
    
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
    
class InverseKineamtics(ForwardKinematics):
    def __init__(self):
        super().__init__()


    def run(self):
        # to get the kinematic chain with the joints and the corresponding parameters, we use the urdf parser
        # documentation of urdf_parser_py see http://wiki.ros.org/urdfdom_py
        robot = URDF.from_parameter_server()
        root = robot.get_root()
        tip = "tool0"
        joint_names = robot.get_chain(root, tip, joints=True, links=False, fixed=False)
        # the properties of a given joint / link can be obtained with the joint_map
        # see http://wiki.ros.org/urdf/XML/joint

        joint_angles = [0,-3.14/2,-3.14/2,0,0,3.14/4] # in radians

        # create the joint state messages
        js = JointState()
        js.name = joint_names      

        # input_vec = [0.3, 0.6, 0.1, 0, 0, 0]# x,y,z,roll,pitch,yaw
        # end_pose = self.get_homogeneous_matrix(input_vec)

        end_joint_angles = [0,-3.14/2,3.14/5,0,0,0] # in radians
        end_pose = self.calculate_forward_kinematics(end_joint_angles)

        target_joint_angles = self.calculate_inverse_kinematics(joint_angles, end_pose)
        js.position = target_joint_angles

        calculated_end_pose = self.calculate_forward_kinematics(target_joint_angles)
        rospy.logerr("target joint angles: {}".format(target_joint_angles))
        rospy.logerr("desired end pose: {}".format(end_pose))
        rospy.logerr("calculated end pose: {}".format(calculated_end_pose))

        if calculated_end_pose is not None:
            target_pose_message = self.get_pose_message_from_matrix(calculated_end_pose)
        else:
            rospy.loginfo("error, no target pose calculated, use identity matrix instead")
            target_pose_message = self.get_pose_message_from_matrix(np.identity(4))

        # publish the joint state values and the target pose
        while not rospy.is_shutdown():
            self.joint_state_publisher.publish(js)
            self.pose_publisher.publish(target_pose_message)
            self.rate.sleep()

    def get_homogeneous_matrix(self, xyzrpy):
        x, y, z, roll, pitch, yaw = xyzrpy
        # to implement, should return a 4x4 homogeneous matrix that corresponds to the pose of the end effector
        # create the homogeneous matrix
        R = np.array([[np.cos(yaw)*np.cos(pitch), np.cos(yaw)*np.sin(pitch)*np.sin(roll) - np.sin(yaw)*np.cos(roll), np.cos(yaw)*np.sin(pitch)*np.cos(roll) + np.sin(yaw)*np.sin(roll)],
                      [np.sin(yaw)*np.cos(pitch), np.sin(yaw)*np.sin(pitch)*np.sin(roll) + np.cos(yaw)*np.cos(roll), np.sin(yaw)*np.sin(pitch)*np.cos(roll) - np.cos(yaw)*np.sin(roll)],
                      [-np.sin(pitch), np.cos(pitch)*np.sin(roll), np.cos(pitch)*np.cos(roll)]])
        T = np.array([[R[0, 0], R[0, 1], R[0, 2], x],
                      [R[1, 0], R[1, 1], R[1, 2], y],
                      [R[2, 0], R[2, 1], R[2, 2], z],
                      [0, 0, 0, 1]])
        return T

    def calculate_inverse_kinematics(self, joint_angles, end_pose):
        # to implement, should return a list of joint angles that correspond to the given end effector pose
        learning_rate = 0.001
        for idx in range(10000):
            current_position, current_rotation = self.get_position_and_rotation(joint_angles)

            target_position = end_pose[:3, 3]
            target_rotation = end_pose[:3, :3]
            
            pos_error = target_position - current_position
            
            # Orientation error: Skew-symmetric error matrix for rotations
            rotation_error = 0.5 * (np.cross(current_rotation[:, 0], target_rotation[:, 0]) +
                                    np.cross(current_rotation[:, 1], target_rotation[:, 1]) +
                                    np.cross(current_rotation[:, 2], target_rotation[:, 2]))
            
            if idx%10 == 0:
                #rospy.logerr("Iteration: {}, Position error: {}".format(idx, pos_error))
                rospy.logerr("Iteration: {}, Rotation error: {}".format(idx, rotation_error))
            
            # Termination condition for both position and orientation
            if np.linalg.norm(pos_error) < 0.01 and np.linalg.norm(rotation_error) < 0.01:
                rospy.logerr("Breaking at iteration: {}".format(idx))
                break
            
            # Jacobian approximation for small perturbations
            delta_theta = 0.1
            J = np.zeros((6, 6))  # Jacobian for position (3) + orientation (3)
            
            # Partial derivatives by finite differences for both position and orientation
            for i in range(6):
                # Perturb joint angle i
                temp_thetas = joint_angles.copy()
                temp_thetas[i] += delta_theta
                pos_delta, rot_delta = self.get_position_and_rotation(temp_thetas)
                
                # Position partial derivative for joint i
                J[:3, i] = (pos_delta - current_position) / delta_theta
                
                # Orientation partial derivative for joint i (rotation error approximation)
                rot_error_delta = 0.5 * (np.cross(rot_delta[:, 0], target_rotation[:, 0]) +
                                        np.cross(rot_delta[:, 1], target_rotation[:, 1]) +
                                        np.cross(rot_delta[:, 2], target_rotation[:, 2]))
                J[3:, i] = (rot_error_delta - rotation_error) / delta_theta
            
            # Error vector for position and orientation
            error = np.hstack((pos_error, rotation_error))
            
            # Update the joint angles using the pseudo-inverse of the Jacobian
            # Update the joint angles using the pseudo-inverse of the Jacobian
            theta_update, _, _, _ = np.linalg.lstsq(J, error, rcond=None)
            joint_angles += learning_rate * theta_update

        return joint_angles

        


if __name__ == '__main__':
    fk = InverseKineamtics()
    fk.run()