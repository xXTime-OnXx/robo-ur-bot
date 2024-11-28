#! /usr/bin/env python3

import numpy as np

from forward_kinematics import ForwardKinematics

class InverseKinematics:
    def __init__(self):
        self.fk = ForwardKinematics()
        self.max_iterations = 100
        self.tolerance = 1e-3
    
    def inverse_kinematics(self, target_pose, initial_angles):
        """
        Löst die IK für eine gegebene Zielpose.
        """
        angles = np.array(initial_angles)
        
        for iteration in range(self.max_iterations):
            # Vorwärtskinematik berechnen
            current_pose = self.fk.calculate(angles)
            
            # Fehler berechnen (Positionsteil)
            position_error = target_pose[:3, 3] - current_pose[:3, 3]
            error_norm = np.linalg.norm(position_error)
            
            if error_norm < self.tolerance:
                print(f"Converged in {iteration} iterations")
                return angles
            
            # Jacobi-Matrix berechnen
            J = self.calculate_jacobian(angles)
            
            # Pseudoinverse der Jacobi-Matrix
            J_pseudo = np.linalg.pinv(J)
            
            # Delta der Gelenkwinkel berechnen
            delta_angles = J_pseudo @ position_error
            
            # Gelenkwinkel aktualisieren
            angles += delta_angles
        
        print("Failed to converge")
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

    def calculate_jacobian(self, joint_angles):
        """
        Numerisch berechnete Jacobi-Matrix.
        """
        delta = 1e-6
        n = len(joint_angles)
        jacobian = np.zeros((6, n))  # 3 für Position, 3 für Orientierung
        
        # Berechnung der Vorwärtskinematik für aktuelle Gelenkwinkel
        current_pose = self.fk.calculate(joint_angles)
        
        for i in range(n):
            perturbed_angles = joint_angles.copy()
            perturbed_angles[i] += delta
            perturbed_pose = self.fk.calculate(perturbed_angles)
            
            # Position und Orientierung ableiten
            position_diff = (perturbed_pose[:3, 3] - current_pose[:3, 3]) / delta
            # Orientierung hier weggelassen oder separat behandelt
            
            jacobian[:3, i] = position_diff  # Positionsteil
            
        return jacobian
    
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
    
    target_angles_deg = [6.56, -73.19, 60.39, -62.35, -1.13, 253.81]
    target_angles = np.deg2rad(target_angles_deg)
    target_pose = fk.calculate(target_angles)
    
    random_angles_deg = ik.randomize_joint_angles()
    random_angles = np.deg2rad(random_angles_deg)
    
    joint_angles = ik.inverse_kinematics(target_pose, random_angles)
    if joint_angles is not None:
        print("Calculated Joint Angles:", np.rad2deg(joint_angles))
    else:
        print("No solution found for the given pose.")
