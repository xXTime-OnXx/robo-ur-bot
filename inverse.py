#!/usr/bin/env python3

import rospy
import numpy as np
import math
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from urdf_parser_py.urdf import URDF

class ForwardKinematics:
    def __init__(self): 
        # Initialize the ROS node
        rospy.init_node('my_joint_state_publisher')
        self.rate = rospy.Rate(10)  # 10 Hz
        rospy.sleep(1.0)

        # Publishers
        self.joint_state_publisher = rospy.Publisher('my_joint_states', JointState, queue_size=10)
        self.pose_publisher = rospy.Publisher('my_pose', PoseStamped, queue_size=10)

        # Load the robot URDF from the ROS parameter server
        self.robot = URDF.from_parameter_server()

        # Get the root link and the tip (end effector) link
        self.root = self.robot.get_root()
        self.tip = "tool0"  # Specify the end effector (tip) link

        # Get the chain of joints between the root and the tip
        self.joint_names = self.robot.get_chain(self.root, self.tip, joints=True, links=False, fixed=False)

    def run(self):
        # Move all joints to default position 0 degrees
        zero_joint_angles = [0.0] * len(self.joint_names)  # Use floats
        # Create joint state message with zero positions
        zero_js = JointState()
        zero_js.name = self.joint_names
        zero_js.position = zero_joint_angles

        # Publish zero joint states for some duration
        rospy.loginfo("Moving joints to zero positions.")
        zero_duration = rospy.Duration(2.0)  # 2 seconds
        start_time = rospy.Time.now()
        while rospy.Time.now() - start_time < zero_duration and not rospy.is_shutdown():
            zero_js.header.stamp = rospy.Time.now()
            self.joint_state_publisher.publish(zero_js)
            self.rate.sleep()

        # Calculate and log the end-effector position at zero joint angles
        end_effector_pose = self.calculate_forward_kinematics(zero_joint_angles)
        if end_effector_pose is not None:
            position = end_effector_pose[:3, 3]
            rospy.loginfo(f"End-effector position at zero joint angles: x={position[0]:.4f}, y={position[1]:.4f}, z={position[2]:.4f}")
        else:
            rospy.logerr("Error calculating end-effector pose at zero joint angles.")
            return

        # Perform inverse kinematics and move to the desired position
        self.move_to_position([1.0, 1.0, 1.0])  # Example desired position

        # Continue publishing the final position
        rospy.loginfo("Reached target joint positions.")
        while not rospy.is_shutdown():
            zero_js.header.stamp = rospy.Time.now()
            self.joint_state_publisher.publish(zero_js)
            self.rate.sleep()

    def calculate_forward_kinematics(self, joint_positions):
        # Start with an identity matrix
        end_effector_pose = np.identity(4)

        # Loop over each joint and calculate its transformation
        for i, joint_name in enumerate(self.joint_names):
            # Get joint information
            joint = self.robot.joint_map[joint_name]
            joint_angle = joint_positions[i]
            joint_origin = joint.origin
            joint_axis = joint.axis

            # Get transformation matrix for this joint
            transformation_matrix = self.get_transformation_matrix_from_urdf(joint_origin, joint_angle, joint_axis)

            # Update the end-effector pose
            end_effector_pose = np.dot(end_effector_pose, transformation_matrix)

        return end_effector_pose

    def get_transformation_matrix_from_urdf(self, origin, angle, axis):
        # Translation
        translation = np.array([origin.xyz[0], origin.xyz[1], origin.xyz[2]])

        # Rotation from origin
        rotation_rpy = origin.rpy  # Roll, pitch, yaw
        rotation_from_origin = self.rotation_matrix_from_rpy(rotation_rpy)

        # Rotation from joint
        rotation_from_joint = self.rotation_matrix_from_axis_angle(axis, angle)

        # Total rotation
        rotation = rotation_from_origin.dot(rotation_from_joint)

        # Build the transformation matrix
        transformation_matrix = np.identity(4)
        transformation_matrix[:3, :3] = rotation
        transformation_matrix[:3, 3] = translation

        return transformation_matrix

    def rotation_matrix_from_rpy(self, rpy):
        """Compute rotation matrix from roll, pitch, yaw angles."""
        roll, pitch, yaw = rpy
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz.dot(Ry).dot(Rx)

    def rotation_matrix_from_axis_angle(self, axis, angle):
        # Compute rotation matrix using axis-angle formula
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)  # Normalize the axis vector
        ux, uy, uz = axis
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        one_minus_cos = 1 - cos_angle

        R = np.array([
            [cos_angle + ux**2 * one_minus_cos,
             ux * uy * one_minus_cos - uz * sin_angle,
             ux * uz * one_minus_cos + uy * sin_angle],
            [uy * ux * one_minus_cos + uz * sin_angle,
             cos_angle + uy**2 * one_minus_cos,
             uy * uz * one_minus_cos - ux * sin_angle],
            [uz * ux * one_minus_cos - uy * sin_angle,
             uz * uy * one_minus_cos + ux * sin_angle,
             cos_angle + uz**2 * one_minus_cos]
        ])
        return R

    def get_pose_message_from_matrix(self, matrix):
        """Return PoseStamped message from homogeneous matrix."""
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "base"
        pose_stamped.header.stamp = rospy.Time.now()
        pose = pose_stamped.pose
        pose.position.x = matrix[0, 3]
        pose.position.y = matrix[1, 3]
        pose.position.z = matrix[2, 3]

        # Convert rotation matrix to quaternion
        q = self.rotation_matrix_to_quaternion(matrix[:3, :3])
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        return pose_stamped

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion."""
        q = np.empty((4,))
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[3] = 0.25 / s
            q[0] = (R[2, 1] - R[1, 2]) * s
            q[1] = (R[0, 2] - R[2, 0]) * s
            q[2] = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                q[3] = (R[2, 1] - R[1, 2]) / s
                q[0] = 0.25 * s
                q[1] = (R[0, 1] + R[1, 0]) / s
                q[2] = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                q[3] = (R[0, 2] - R[2, 0]) / s
                q[0] = (R[0, 1] + R[1, 0]) / s
                q[1] = 0.25 * s
                q[2] = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                q[3] = (R[1, 0] - R[0, 1]) / s
                q[0] = (R[0, 2] + R[2, 0]) / s
                q[1] = (R[1, 2] + R[2, 1]) / s
                q[2] = 0.25 * s
        return q

    def move_to_position(self, p_desired):
        # Perform inverse kinematics to find the required joint angles
        q_initial = [0.0] * len(self.joint_names)  # Initial guess for joint angles
        q_solution = self.inverse_kinematics(p_desired, q_initial)

        # Log the solution joint angles
        rospy.loginfo(f"Solution joint angles: {q_solution}")

        # Create a JointState message with the solution joint angles
        js = JointState()
        js.name = self.joint_names
        js.position = q_solution

        # Publish the joint states to move the robot
        rospy.loginfo("Moving to the calculated joint angles...")
        move_duration = rospy.Duration(2.0)  # 2 seconds
        start_time = rospy.Time.now()
        while rospy.Time.now() - start_time < move_duration and not rospy.is_shutdown():
            js.header.stamp = rospy.Time.now()
            self.joint_state_publisher.publish(js)
            self.rate.sleep()

        rospy.loginfo("Reached calculated joint positions.")

    def forward_kinematics(self, q):
        end_effector_pose = np.identity(4)
        for i, joint_name in enumerate(self.joint_names):
            joint = self.robot.joint_map[joint_name]
            joint_angle = q[i]
            joint_origin = joint.origin
            joint_axis = joint.axis
            transformation_matrix = self.get_transformation_matrix_from_urdf(joint_origin, joint_angle, joint_axis)
            end_effector_pose = np.dot(end_effector_pose, transformation_matrix)
        position = end_effector_pose[:3, 3]
        return position

    def jacobian(self, q, delta=1e-6):
        J = np.zeros((3, len(q)))  # Initialize a 3xN Jacobian matrix for 3D position control
        for i in range(len(q)):
            q1 = np.copy(q)
            q2 = np.copy(q)
            q1[i] -= delta / 2.0
            q2[i] += delta / 2.0
            pos1 = self.forward_kinematics(q1)
            pos2 = self.forward_kinematics(q2)
            J[:, i] = (pos2 - pos1) / delta
        return J

    def inverse_kinematics(self, p_desired, q_initial, tol=1e-6, max_iter=100):
        q = np.array(q_initial)
        for _ in range(max_iter):
            p_current = self.forward_kinematics(q)
            error = np.array(p_desired) - p_current
            if np.linalg.norm(error) < tol:
                break
            J = self.jacobian(q)
            q = q + np.linalg.pinv(J).dot(error)  # Use pseudo-inverse for numerical stability
        return q

if __name__ == '__main__':
    fk = ForwardKinematics()
    fk.run()
