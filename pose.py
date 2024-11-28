#! /usr/bin/env python3

class Pose:
    
    def __init__(self, x, y, z, yaw, pitch, roll):
         self._x = x
         self._y = y
         self._z = z
         self._yaw = yaw
         self._pitch = pitch
         self._roll = roll
         
    def __str__(self):
        return (
            f"Position:\n"
            f"  x: {self._x}\n"
            f"  y: {self._y}\n"
            f"  z: {self._z}\n"
            f"Orientation:\n"
            f"  yaw: {self._yaw}\n"
            f"  pitch: {self._pitch}\n"
            f"  roll: {self._roll}\n"
        )