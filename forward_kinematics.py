#! /usr/bin/env python3

import numpy as np
import math

from pose import Pose

class ForwardKinematics:

    def __init__(self):
        self._OFFSET_JOINT_1 = 0.15185 # height of base
        self._OFFSET_JOINT_2 = -0.24355 # length of second joint
        self._OFFSET_JOINT_3 = -0.2132 # length of third joint
        self._OFFSET_JOINT_4 = -0.13105 # length of fourth joint
        self._OFFSET_JOINT_5 = -0.08535 # length of fifth joint
        self._OFFSET_JOINT_6 = -0.0921 # length of sixth joint

    def run(self, joint_angles):
        end_effector_matrix = self.calculate(joint_angles)
        
        print(self.get_pose_map_from_matrix(end_effector_matrix))
        
 
    def calculate(self, joint_angles):
        '''
            Params:
                - 'joint_positions': joint positions in rad
                
            Returns: cords and pose of the tool
        '''
        TR1 = self.R_z(joint_angles[0]) # rotate around z-axis by joint angle
        TM1 = self.E(z = self._OFFSET_JOINT_1) # move by height of base

        TR2 = self.R_y(joint_angles[1])
        TM2 = self.E(x = self._OFFSET_JOINT_2) 
        
        TR3 = self.R_y(joint_angles[2])
        TM3 = self.E(x = self._OFFSET_JOINT_3)
        
        TR4 = self.R_y(joint_angles[3])
        TM4 = self.E(y = self._OFFSET_JOINT_4)
        
        TR5 = self.R_z(joint_angles[4])
        TM5 = self.E(z = self._OFFSET_JOINT_5)
        
        TR6 = self.R_y(joint_angles[5])
        TM6 = self.E(y = self._OFFSET_JOINT_6)
        

        T1 = TR1 @ TM1 
        T2 = T1 @ TR2 @ TM2 
        T3 = T2 @ TR3 @ TM3
        T4 = T3 @ TR4 @ TM4 
        T5 = T4 @ TR5 @ TM5
        TR = T5 @ TR6 @ TM6
        
        return TR
    
        
    def R_x(self, theta, x = 0, y = 0, z = 0):
        return np.array([
            [1,         0,              0,              x],
            [0,         np.cos(theta),  np.sin(theta),  y],
            [0,         -np.sin(theta), np.cos(theta),  z],
            [0,         0,              0,              1]
        ])
        
    def R_y(self, theta, x = 0, y = 0, z = 0):
        return np.array([
            [np.cos(theta),     0,          -np.sin(theta),     x],
            [0,                 1,          0,                  y],
            [np.sin(theta),     0,          np.cos(theta),      z],
            [0,                 0,          0,                  1]
        ])
        
    def R_z(self, theta, x = 0, y = 0, z = 0):
        return np.array([
            [np.cos(theta),     -np.sin(theta),     0,      x],
            [np.sin(theta),     np.cos(theta),      0,      y],
            [0,                 0,                  1,      z],
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
    def get_pose_map_from_matrix(self, matrix):
        """Return pose map from homogeneous matrix
        matrix : homogeneous matrix 4x4
        """
        q = self.get_quaternion_from_matrix(matrix)

        return Pose(matrix[0][3], matrix[1][3], matrix[2][3], q[0], q[1], q[2], q[3])
    

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
    joint_angles = [0, 90, 0, -90, 0, 88] # in radians
    joint_angles_in_rad = np.deg2rad(joint_angles)
    
    fk = ForwardKinematics()
    fk.run(joint_angles_in_rad)
    
    
