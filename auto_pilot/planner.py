import os
from collections import deque

import numpy as np


DEBUG = False


class Plotter(object):
    def __init__(self, size, gps=False):
        self.size = size
        self.clear()
        self.title = str(self.size)
        self.scale = 100000

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = self.scale * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

        self.mean = np.array([49.0, 8.0])
        self.scale = np.array([111324.60662786, 73032.1570362])
        self.mean = np.array([0, 0])
        self.scale = np.array([1, 1])

        self.debug = Plotter(debug_size)

    def set_route(self, global_plan_gps, global_plan_world_coord):
        self.route.clear()
        self.route_size = len(global_plan_gps)

        for (gps_pos, cmd), (pos, _) in zip(global_plan_gps, global_plan_world_coord):
            gps_pos = np.array([gps_pos['lat'], gps_pos['lon']])
            gps_pos -= self.mean
            gps_pos *= self.scale

            self.route.append((gps_pos, cmd, pos))

    def route_completion(self):
        return 1 - len(self.route)/self.route_size

    def route_completed(self):
        return len(self.route) == 2

    def run_step(self, gps):
        self.debug.clear()

        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        self.debug.dot(gps, self.route[0][0], (0, 255, 0))
        self.debug.dot(gps, self.route[1][0], (255, 0, 0))
        self.debug.dot(gps, gps, (0, 0, 255))
        self.debug.show()

        return self.route[1]