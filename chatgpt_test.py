import numpy as np
import rospy
import socket

class RealRobotArm:
    def __init__(self):
        host = rospy.get_param("robot_ip")
        port_ur = 30002
        port_gripper = 63352

        rospy.init_node('my_real_robot')
        rospy.sleep(3.0)        

        # Create socket connection to robot arm and gripper
        self.socket_ur = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_ur.connect((host, port_ur))
        self.socket_gripper = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_gripper.connect((host, port_gripper))
        
        # Activate the gripper
        self.socket_gripper.sendall(b'SET ACT 1\n')

    def send_joint_command(self, joint_angles):
        values = ', '.join(['{:.2f}'.format(i) if type(i) == float else str(i) for i in joint_angles])
        self.socket_ur.send(str.encode("movej([" + values + "])\n"))

    def send_gripper_command(self, value):
        if 0 <= value <= 255:
            command = 'SET POS ' + str(value) + '\n'
            self.socket_gripper.send(str.encode(command))
            # Make the gripper move
            self.socket_gripper.send(b'SET GTO 1\n')

    def close_connection(self):
        self.socket_ur.close()
        self.socket_gripper.close()

class InverseKinematics:
    def __init__(self):
        # Define robot's DH parameters (example values; replace with your robot's specific parameters)
        self.a = [0, 0.5, 0.5, 0, 0, 0]  # Link lengths
        self.alpha = [0, 0, 0, np.pi/2, -np.pi/2, 0]  # Link twists
        self.d = [0.4, 0, 0, 0.4, 0, 0.2]  # Link offsets
        self.theta_offset = [0, 0, 0, 0, 0, 0]  # Joint angle offsets

    def calculate_inverse_kinematics(self, target_position):
        """
        Solves inverse kinematics to find joint angles for the target position.
        Arguments:
            target_position: [x, y, z] coordinates of the end effector.
        Returns:
            joint_angles: Calculated joint angles for the robot.
        """
        x, y, z = target_position
        
        # Placeholder algorithm: assumes a specific solution approach.
        # Replace with analytical or numerical methods as per your robot's kinematics.
        joint_angles = [0, 0, 0, 0, 0, 0]  # Initialize

        # Example simple planar arm calculation for 3 joints
        r = np.sqrt(x**2 + y**2)  # Planar distance to target
        z_offset = z - self.d[0]  # Vertical distance to target

        # Solve for the first two joints (assuming planar motion for simplicity)
        cos_theta2 = (r**2 + z_offset**2 - self.a[1]**2 - self.a[2]**2) / (2 * self.a[1] * self.a[2])
        sin_theta2 = np.sqrt(1 - cos_theta2**2)  # Elbow-up solution

        joint_angles[1] = np.arctan2(sin_theta2, cos_theta2)
        joint_angles[0] = np.arctan2(y, x) - np.arctan2(self.a[2] * sin_theta2, self.a[1] + self.a[2] * cos_theta2)
        joint_angles[2] = np.arctan2(z_offset, r) - joint_angles[1]

        # Return calculated angles
        return joint_angles

if __name__ == '__main__':
    robot = RealRobotArm()
    ik_solver = InverseKinematics()

    # Target position (replace with desired coordinates)
    target_position = [0.4, 0.2, 0.3]

    # Calculate joint angles
    joint_angles = ik_solver.calculate_inverse_kinematics(target_position)

    # Send joint commands to robot
    robot.send_joint_command(joint_angles)

    # Close connection
    robot.close_connection()