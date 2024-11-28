#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from urdf_parser_py.urdf import URDF

class InverseKinematics:
    def __init__(self):
        self.publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)
        self.joint_state_publisher = rospy.Publisher('my_joint_states', JointState, queue_size=10) 
        self.pose_publisher = rospy.Publisher('my_pose', PoseStamped, queue_size=10)
        rospy.init_node("inverse_kinematics_node")
        self.rate = rospy.Rate(10)
        rospy.sleep(1.0)

        # Publishers
        self.joint_state_publisher = rospy.Publisher("my_joint_states", JointState, queue_size=10)
        self.pose_publisher = rospy.Publisher("my_pose", PoseStamped, queue_size=10)

        # Load the robot URDF from the ROS parameter server
        self.robot = URDF.from_parameter_server()
        self.root = self.robot.get_root()
        self.tip = "tool0"  # Specify the end-effector (tip) link
        self.joint_names = self.robot.get_chain(self.root, self.tip, joints=True, links=False, fixed=False)
        
    def update_joint_state(self, joint_angles):
        """
        Publish updated joint state to the /joint_states topic.
        """
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = rospy.Time.now()
        
        # Joint names must match those in your robot URDF or the robot model.
        joint_state.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # Replace with actual joint names
        joint_state.position = joint_angles  # Set joint angles from the inverse kinematics calculation
        
        # Publish the joint state message
        self.publisher.publish(joint_state)

    def run(self):
        # Define a target pose (position only, for simplicity)
        target_pose = np.eye(4)
        target_pose[:3, 3] = [0.5, 0.2, 0.3]  # Example target position

        rospy.loginfo(f"Target position: {target_pose[:3, 3]}")

        # Perform inverse kinematics to calculate joint angles
        try:
            joint_angles = self.calculate_inverse_kinematics(target_pose)
            rospy.loginfo(f"Calculated joint angles: {joint_angles}")

            # Move the robot to the calculated joint angles
            self.move_to_joint_angles(joint_angles)
        except Exception as e:
            rospy.logerr(f"Error in inverse kinematics: {e}")

    def calculate_forward_kinematics(self, joint_angles):
        """Calculate the forward kinematics for the current joint angles."""
        end_effector_pose = np.identity(4)
        for i, joint_name in enumerate(self.joint_names):
            joint = self.robot.joint_map[joint_name]
            joint_angle = joint_angles[i]
            joint_origin = joint.origin
            joint_axis = joint.axis
            transformation_matrix = self.get_transformation_matrix_from_urdf(joint_origin, joint_angle, joint_axis)
            end_effector_pose = np.dot(end_effector_pose, transformation_matrix)
        return end_effector_pose

    def calculate_inverse_kinematics(self, target_pose):
        """Calculate joint angles to reach the target pose using Newton's method."""
        joint_angles = np.zeros(len(self.joint_names))  # Initial guess
        for _ in range(100):  # Max iterations
            current_pose = self.calculate_forward_kinematics(joint_angles)
            error = target_pose[:3, 3] - current_pose[:3, 3]
            if np.linalg.norm(error) < 1e-3:  # Convergence threshold
                break
            J = self.calculate_jacobian(joint_angles)  # Numerical Jacobian
            delta_theta = np.linalg.pinv(J) @ error
            joint_angles += delta_theta
        return joint_angles

    def calculate_jacobian(self, joint_angles):
        """Calculate the numerical Jacobian for the robot."""
        epsilon = 1e-6
        J = np.zeros((3, len(joint_angles)))
        for i in range(len(joint_angles)):
            perturbed_angles = joint_angles.copy()
            perturbed_angles[i] += epsilon
            perturbed_pose = self.calculate_forward_kinematics(perturbed_angles)
            current_pose = self.calculate_forward_kinematics(joint_angles)
            J[:, i] = (perturbed_pose[:3, 3] - current_pose[:3, 3]) / epsilon
        return J

    def move_to_joint_angles(self, joint_angles):
        """Move the robot to the calculated joint angles and visualize the movement."""
        steps = 100  # Number of steps for smooth transition
        step_duration = 0.05  # Time for each step in seconds
        initial_joint_angles = np.zeros(len(self.joint_names))  # Initial joint angles (starting position)

        for step in range(steps + 1):
            if rospy.is_shutdown():
                break
            # Interpolate joint angles to move smoothly from initial to target
            fraction = step / steps
            current_joint_angles = [
                initial + fraction * (target - initial)
                for initial, target in zip(initial_joint_angles, joint_angles)
            ]
            
            # Publish the joint states to control the robot's actuators
            js = JointState()
            js.name = self.joint_names
            js.position = current_joint_angles
            js.header.stamp = rospy.Time.now()
            self.joint_state_publisher.publish(js)
            
            # Calculate forward kinematics to get the current pose of the robot's end effector
            current_pose = self.calculate_forward_kinematics(current_joint_angles)
            # Visualize the current pose
            target_pose_message = self.get_pose_message_from_matrix(current_pose)
            self.pose_publisher.publish(target_pose_message)
            
            rospy.sleep(step_duration)  # Sleep to control the speed of motion





    def get_transformation_matrix_from_urdf(self, origin, angle, axis):
        translation = np.array([origin.xyz[0], origin.xyz[1], origin.xyz[2]])
        rotation_rpy = origin.rpy
        rotation_from_origin = self.rotation_matrix_from_rpy(rotation_rpy)
        rotation_from_joint = self.rotation_matrix_from_axis_angle(axis, angle)
        rotation = rotation_from_origin.dot(rotation_from_joint)
        transformation_matrix = np.identity(4)
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = translation
        return transformation_matrix

    def rotation_matrix_from_rpy(self, rpy):
        roll, pitch, yaw = rpy
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        return Rz.dot(Ry).dot(Rx)

    def rotation_matrix_from_axis_angle(self, axis, angle):
        axis = np.array(axis) / np.linalg.norm(axis)
        ux, uy, uz = axis
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle
        return np.array([
            [cos_angle + ux**2 * one_minus_cos, ux*uy*one_minus_cos - uz*sin_angle, ux*uz*one_minus_cos + uy*sin_angle],
            [uy*ux*one_minus_cos + uz*sin_angle, cos_angle + uy**2 * one_minus_cos, uy*uz*one_minus_cos - ux*sin_angle],
            [uz*ux*one_minus_cos - uy*sin_angle, uz*uy*one_minus_cos + ux*sin_angle, cos_angle + uz**2 * one_minus_cos]
        ])

    def get_pose_message_from_matrix(self, matrix):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.pose.position.x = matrix[0, 3]
        pose_stamped.pose.position.y = matrix[1, 3]
        pose_stamped.pose.position.z = matrix[2, 3]
        return pose_stamped

if __name__ == "__main__":
    ik = InverseKinematics()
    ik.run()
