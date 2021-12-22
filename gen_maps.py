import queue
from pathlib import Path

from PIL import Image, ImageDraw, ImageColor
import numpy as np

import carla
from carla import ColorConverter as cc

from auto_pilot.route_parser import parse_routes_file
from auto_pilot.route_manipulation import interpolate_trajectory


class Camera(object):
    def __init__(self, world, w, h, fov, x, y, z, pitch, yaw):
        bp_library = world.get_blueprint_library()
        camera_bp = bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(w))
        camera_bp.set_attribute('image_size_y', str(h))
        camera_bp.set_attribute('fov', str(fov))

        loc = carla.Location(x=x, y=y, z=z)
        rot = carla.Rotation(pitch=pitch, yaw=yaw)
        transform = carla.Transform(loc, rot)

        self.queue = queue.Queue()

        self.camera = world.spawn_actor(camera_bp, transform)
        self.camera.listen(self.queue.put)

    def get(self):
        image = None

        while image is None or self.queue.qsize() > 0:
            image = self.queue.get()

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        return array

    def __del__(self):
        pass
        # self.camera.destroy()

        # with self.queue.mutex:
        #     self.queue.queue.clear()


def process_img(image):
    image.convert(cc.Raw)
    image.save_to_disk('_out/%08d' % image.frame_number)


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0

    world.apply_settings(settings)


host = 'localhost'
port = 2000
client = carla.Client(host, port)
client.set_timeout(30.0)
world = client.get_world()

set_sync_mode(client, True)

routes_file = Path('data/routes_training.xml')
routes = parse_routes_file(routes_file)
for route_idx in range(len(routes)):
    route = routes[route_idx]
    world = client.load_world(route['town'])
    trajectory = route['trajectory']

    # set_sync_mode(_client, True)

    global_plan_gps, global_plan_world_coord = interpolate_trajectory(world, trajectory)
    max_x = global_plan_world_coord[0][0].location.x
    min_x = global_plan_world_coord[0][0].location.x
    max_y = global_plan_world_coord[0][0].location.y
    min_y = global_plan_world_coord[0][0].location.y
    for trajectory_point in global_plan_world_coord:
        point_x = trajectory_point[0].location.x
        point_y = trajectory_point[0].location.y
        if point_x > max_x:
            max_x = point_x
        elif point_x < min_x:
            min_x = point_x

        if point_y > max_y:
            max_y = point_y
        elif point_y < min_y:
            min_y = point_y

    image_width = 4096
    image_height = 4096
    camera_fov = 90
    camera_x = (max_x + min_x) / 2
    camera_y = (max_y + min_y) / 2
    traj_radius = max(max_x - camera_x, max_y - camera_y)
    camera_z = 1.2 * np.tan((90 - camera_fov / 2) * np.pi / 180) * traj_radius
    camera = Camera(world, image_width, image_height, camera_fov, camera_x, camera_y, camera_z, -90, 0)
    world.tick()
    result = camera.get()
    image = Image.fromarray(result)
    draw = ImageDraw.Draw(image)
    last_point = global_plan_world_coord[0][0].location
    meters_to_pixel_x = image_width / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)
    meters_to_pixel_y = -image_height / (2 * np.tan(camera_fov * np.pi / 180 / 2) * camera_z)
    last_point_x = meters_to_pixel_x * (last_point.y - camera_y) + image_width / 2
    last_point_y = meters_to_pixel_y * (last_point.x - camera_x) + image_height / 2
    radius = 20
    n_points = len(global_plan_world_coord)
    rgb_diff = -255 / n_points
    rgb = 255
    draw.ellipse([last_point_x - radius, last_point_y - radius, last_point_x + radius, last_point_y + radius], fill=(255, int(rgb), 0))
    rgb += rgb_diff
    for point_idx in range(1, len(global_plan_world_coord)):
        point_loc = global_plan_world_coord[point_idx][0].location
        point_x = meters_to_pixel_x * (point_loc.y - camera_y) + image_width / 2
        point_y = meters_to_pixel_y * (point_loc.x - camera_x) + image_height / 2
        draw.line((last_point_x, last_point_y, point_x, point_y), width=20, fill=(255, int(rgb), 0))
        rgb += rgb_diff
        last_point_x = point_x
        last_point_y = point_y
    draw.ellipse([last_point_x - radius, last_point_y - radius, last_point_x + radius, last_point_y + radius], fill=(255, int(rgb), 0))

    image.save('trajs/route_%02d.png' % route_idx)