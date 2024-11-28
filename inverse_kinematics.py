#! /usr/bin/env python3

import numpy as np
import math

from forward_kinematics import ForwardKinematics

class InverseKinematics:
    def __init__(self):
        self.forward_kinematics = ForwardKinematics()
    
    def inverse_kinematics(self, target_pose, initial_angles, tolerance=1e-6, max_iterations=1000):
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
            current_pose = self.forward_kinematics.calculate(theta)
            delta = self.calculate_error(target_pose, current_pose)
            
            # Abbruchkriterium prüfen
            if np.linalg.norm(delta) < tolerance:
                return theta

            # Jacobi-Matrix berechnen
            J = self.calculate_jacobian(theta)

            # Gelenkwinkel aktualisieren
            try:
                theta += np.linalg.pinv(J) @ delta
            except np.linalg.LinAlgError:
                print("Jacobian ist singulär")
                return None
        print("Maximale Iterationen erreicht")
        return None

    def calculate_error(self, target_pose, current_pose):
        """Berechnet die Abweichung zwischen Soll- und Ist-Position."""
        # Positionsfehler
        position_error = target_pose[:3, 3] - current_pose[:3, 3]
        
        # Orientierungsfehler
        target_quaternion = self.forward_kinematics.get_quaternion_from_matrix(target_pose)
        current_quaternion = self.forward_kinematics.get_quaternion_from_matrix(current_pose)
        
        # Quaternion-Differenz (relative Orientierung)
        orientation_error = self.quaternion_error(target_quaternion, current_quaternion)
        
        return np.hstack((position_error, orientation_error))
    
    
    def quaternion_error(self, target_quaternion, current_quaternion):
        """Berechnet den Orientierungsfehler basierend auf Quaternions."""
        q_target = np.array(target_quaternion)
        q_current = np.array(current_quaternion)
        
        # Quaternion-Konjungierte von Ist-Wert
        q_conj = q_current * np.array([1, -1, -1, -1])  # [q0, -q1, -q2, -q3]
        
        # Relativer Quaternion-Fehler
        q_relative = np.array([
            q_target[0] * q_conj[0] - np.dot(q_target[1:], q_conj[1:]),
            *(q_target[0] * q_conj[1:] + q_conj[0] * q_target[1:] + np.cross(q_target[1:], q_conj[1:]))
        ])
        
        # Rückgabe nur der imaginären Teile (Orientierungsvektor)
        return q_relative[1:]

    def calculate_jacobian(self, theta):
        """Numerische Berechnung der Jacobi-Matrix."""
        delta_theta = 1e-5
        J = np.zeros((6, len(theta)))
        
        current_pose = self.forward_kinematics.calculate(theta)
        for i in range(len(theta)):
            theta_perturbed = theta.copy()
            theta_perturbed[i] += delta_theta
            
            # Berechnung der Pose bei perturbiertem Gelenkwinkel
            pose_perturbed = self.forward_kinematics.calculate(theta_perturbed)
            
            # Positionsteil der Jacobi-Matrix
            delta_position = (pose_perturbed[:3, 3] - current_pose[:3, 3]) / delta_theta
            J[:3, i] = delta_position  # Positionsteil
            
            # Orientierungsteil der Jacobi-Matrix
            q_perturbed = self.forward_kinematics.get_quaternion_from_matrix(pose_perturbed)
            q_current = self.forward_kinematics.get_quaternion_from_matrix(current_pose)
            delta_orientation = (self.quaternion_error(q_perturbed, q_current)) / delta_theta
            J[3:, i] = delta_orientation  # Orientierungsteil
        
        return J
    
    def randomize_joint_angles(self):
        """
        Generiert zufällige Gelenkwinkel innerhalb der angegebenen Grenzen.
        
        :param joint_limits: Liste von Tupeln [(min1, max1), (min2, max2), ...] für jedes Gelenk
        :return: Array von Gelenkwinkeln innerhalb der Grenzen
        """
        joint_limits = [
            (-np.pi, np.pi),     # Gelenk 1: volle Drehung
            (-np.pi / 2, np.pi / 2),  # Gelenk 2: Neigung
            (-np.pi / 2, np.pi / 2),  # Gelenk 3: Neigung
            (-np.pi, np.pi),     # Gelenk 4: volle Drehung
            (-np.pi, np.pi),     # Gelenk 5: volle Drehung
            (-np.pi / 2, np.pi / 2)   # Gelenk 6: Endeffektor-Drehung
        ]
        
        random_angles = []
        for min_angle, max_angle in joint_limits:
            angle = np.random.uniform(min_angle, max_angle)
            random_angles.append(angle)
        return np.array(random_angles)


if __name__ == '__main__':
    ik = InverseKinematics()
    fk = ForwardKinematics()
    
    target_angles_deg = [-87.8, -155.27, 6.21, -33.99, 85.44, 0.09]
    target_angles = np.deg2rad(target_angles_deg)
    target_pose = fk.calculate(target_angles)
    

    
    solutions = list()
    for epoch in range(100):
        random_angles_deg = ik.randomize_joint_angles()
        random_angles = np.deg2rad(random_angles_deg)
        
        print("Epoch: " + str(epoch))
        joint_angles = ik.inverse_kinematics(target_pose, random_angles)
        if joint_angles is not None:
            joint_angles = joint_angles % (math.pi * 2)
            print("Calculated Joint Angles:", np.rad2deg(joint_angles))
            solutions.append(joint_angles)
        else:
            print("No solution found for the given pose.")
