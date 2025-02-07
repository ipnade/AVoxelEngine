import pygame as pg
import math
import glm

class Player:
    def __init__(self):
        # Replace Vector3 with glm.vec3
        self.position = glm.vec3(8.0, 70.0, 8.0)
        self.yaw = -90.0
        self.pitch = 0.0
        self.speed = 10.0
        self.sensitivity = 0.1

    def get_direction(self):
        rad_yaw = math.radians(self.yaw)
        rad_pitch = math.radians(self.pitch)
        x = math.cos(rad_yaw) * math.cos(rad_pitch)
        y = math.sin(rad_pitch)
        z = math.sin(rad_yaw) * math.cos(rad_pitch)
        return glm.normalize(glm.vec3(x, y, z))

    def get_right(self):
        # Use glm.cross instead of Vector3.cross
        return glm.normalize(glm.cross(self.get_direction(), glm.vec3(0, 1, 0)))

    def get_left(self):
        return -self.get_right()

    def get_up(self):
        return glm.normalize(glm.cross(self.get_right(), self.get_direction()))

    def process_mouse(self, dx, dy):
        self.yaw += dx * self.sensitivity
        self.pitch -= dy * self.sensitivity
        self.pitch = max(-89.0, min(89.0, self.pitch))

    def process_keyboard(self, keys, dt):
        forward = self.get_direction()
        right = self.get_right()
        left = self.get_left()  # Added left vector for clarity, though -right works too.
        up = glm.vec3(0, 1, 0)
        if keys[pg.K_w]:
            self.position += forward * (self.speed * dt)
        if keys[pg.K_s]:
            self.position -= forward * (self.speed * dt)
        if keys[pg.K_a]:
            self.position += left * (self.speed * dt)
        if keys[pg.K_d]:
            self.position += right * (self.speed * dt)
        if keys[pg.K_SPACE]:
            self.position += up * (self.speed * dt)
        if keys[pg.K_LSHIFT]:
            self.position -= up * (self.speed * dt)