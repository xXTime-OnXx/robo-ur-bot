#! /usr/bin/env python3

class Pose:
    
    def __init__(self, x, y, z, q_w, q_x, q_y, q_z):
         self._x = x
         self._y = y
         self._z = z
         self._q_w = q_w
         self._q_x = q_x
         self._q_y = q_y
         self._q_z = q_z
         
    def __str__(self):
        return (
            f"Position:\n"
            f"  x: {self._x}\n"
            f"  y: {self._y}\n"
            f"  z: {self._z}\n"
            f"Orientation:\n"
            f"  q_w: {self._q_w}\n"
            f"  q_x: {self._q_x}\n"
            f"  q_y: {self._q_y}\n"
            f"  q_z: {self._q_z}\n"
        )