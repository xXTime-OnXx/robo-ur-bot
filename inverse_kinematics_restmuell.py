#! /usr/bin/env python3

import rospy
import numpy as np
import math
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from urdf_parser_py.urdf import URDF

class InverseKinematics:
    def __init__(self):
        rospy.init_node('inverse_kinematics_solver')
        self.rate = rospy.Rate(10)
        self.robot = URDF.from_parameter_server()
        self.root = self.robot.get_root()
        self.tip = "tool0"
        self.joint_names = self.robot.get_chain(self.root, self.tip, joints=True, links=False, fixed=False)
    
    def inverse_kinematics(self, target_pose, initial_angles, forward_kinematics, tolerance=1e-6, max_iterations=100):
        """
        Berechnet die Gelenkwinkel für eine Soll-Position.
        
        :param target_pose: Zielposition (4x4 Homogene Matrix)
        :param initial_angles: Initiale Gelenkwinkel (1x6 Vektor)
        :param forward_kinematics: Funktion zur Berechnung der Vorwärtskinematik
        :param tolerance: Toleranz für die Abweichung
        :param max_iterations: Maximale Anzahl Iterationen
        :return: Gelenkwinkel oder None, wenn keine Lösung gefunden wurde
        """
        theta = np.array(initial_angles)
        for i in range(max_iterations):
            # Ist-Position berechnen
            current_pose = forward_kinematics(theta)
            delta = calculate_error(target_pose, current_pose)
            
            # Abbruchkriterium prüfen
            if np.linalg.norm(delta) < tolerance:
                return theta

            # Jacobi-Matrix berechnen
            J = calculate_jacobian(theta, forward_kinematics)

            # Gelenkwinkel aktualisieren
            try:
                theta = theta - np.linalg.pinv(J) @ delta
            except np.linalg.LinAlgError:
                print("Jacobian ist singulär")
                return None
        print("Maximale Iterationen erreicht")
        return None

    def calculate_error(self, target_pose, current_pose):
        """Berechnet die Abweichung zwischen Soll- und Ist-Position."""
        position_error = target_pose[:3, 3] - current_pose[:3, 3]
        # Orientierung (z. B. als Quaternion oder Roll-Pitch-Yaw)
        orientation_error = ...  # Konvertiere Matrix in Roll-Pitch-Yaw und vergleiche
        return np.hstack((position_error, orientation_error))

    def calculate_jacobian(self, theta, forward_kinematics):
        """Numerische Berechnung der Jacobi-Matrix."""
        delta_theta = 1e-5
        J = np.zeros((6, len(theta)))
        for i in range(len(theta)):
            theta_perturbed = theta.copy()
            theta_perturbed[i] += delta_theta
            pose_perturbed = forward_kinematics(theta_perturbed)
            pose_current = forward_kinematics(theta)
            delta = (pose_perturbed[:3, 3] - pose_current[:3, 3]) / delta_theta
            J[:3, i] = delta  # Positionsteil
            # Orientierungsteil berechnen ...
        return J

if __name__ == '__main__':
    ik = InverseKinematics()
    # Example target pose matrix
    target_pose = np.array([
        [1, 0, 0, -0.3],
        [0, 1, 0, -0.3],
        [0, 0, 1, 0.3],
        [0, 0, 0, 1]
    ])
    random_angles = [0, -1.57, 0, 0, 0, 0]
    
    joint_angles = ik.inverse_kinematics(target_pose, random_angles, )
    if joint_angles:
        print("Calculated Joint Angles:", joint_angles)
    else:
        print("No solution found for the given pose.")
