import math
import numpy as np
import gym
from gym import spaces
import collections
import queue
import time

import carla

from agents.navigation.local_planner import RoadOption

from PIL import Image, ImageDraw
from auto_pilot.route_parser import parse_routes_file
from auto_pilot.route_manipulation import interpolate_trajectory
from auto_pilot.planner import RoutePlanner, Plotter
from auto_pilot.route_manipulation import downsample_route
from auto_pilot.pid_controller import PIDController


VEHICLE_NAME = 'vehicle.lincoln.mkz2017'


def set_sync_mode(client, sync):
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = sync
    settings.fixed_delta_seconds = 1.0 / 10.0

    world.apply_settings(settings)


class Camera(object):
    def __init__(self, world, player, w, h, fov, x, y, z, pitch, yaw, type='rgb'):
        bp = world.get_blueprint_library().find('sensor.camera.%s' % type)
        bp.set_attribute('image_size_x', str(w))
        bp.set_attribute('image_size_y', str(h))
        bp.set_attribute('fov', str(fov))

        loc = carla.Location(x=x, y=y, z=z)
        rot = carla.Rotation(pitch=pitch, yaw=yaw)
        transform = carla.Transform(loc, rot)

        self.type = type
        self.queue = queue.Queue()

        self.camera = world.spawn_actor(bp, transform, attach_to=player)
        self.camera.listen(self.queue.put)

    def get(self):
        image = None

        while image is None or self.queue.qsize() > 0:
            image = self.queue.get()

        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.type == 'semantic_segmentation':
            return array[:, :, 0]

        return array

    def __del__(self):
        self.camera.destroy()

        with self.queue.mutex:
            self.queue.queue.clear()


class GNSS(object):
    def __init__(self, world, player):
        bp = world.get_blueprint_library().find('sensor.other.gnss')

        gnss_location = carla.Location(0, 0, 0)
        gnss_rotation = carla.Rotation(0, 0, 0)
        gnss_transform = carla.Transform(gnss_location, gnss_rotation)

        self.type = type
        self.queue = queue.Queue()

        self.gnss = world.spawn_actor(
            bp, gnss_transform, attach_to=player, attachment_type=carla.AttachmentType.Rigid)
        self.gnss.listen(self.queue.put)
        self.gnss_data = np.array([0, 0])
        self.initialized = False

    def get(self):
        if not self.initialized:
            raw_data = self.queue.get()
            self.gnss_data = np.array([raw_data.latitude, raw_data.longitude])
            self.initialized = True

        while self.queue.qsize() > 0:
            raw_data = self.queue.get()
            self.gnss_data = np.array([raw_data.latitude, raw_data.longitude])

        return self.gnss_data

    def __del__(self):
        self.gnss.destroy()

        with self.queue.mutex:
            self.queue.queue.clear()


class IMU(object):
    def __init__(self, world, player):
        imu_bp = world.get_blueprint_library().find('sensor.other.imu')

        imu_location = carla.Location(0, 0, 0)
        imu_rotation = carla.Rotation(0, 0, 0)
        imu_transform = carla.Transform(imu_location, imu_rotation)

        self.type = type
        self.queue = queue.Queue()

        self.imu = world.spawn_actor(
            imu_bp, imu_transform, attach_to=player, attachment_type=carla.AttachmentType.Rigid)
        self.imu.listen(self.queue.put)
        self.imu_data = np.array([0, 0])
        self.initialized = False

    def get(self):
        if not self.initialized:
            raw_data = self.queue.get()
            self.imu_data = np.array([raw_data.accelerometer.x,
                                      raw_data.accelerometer.y,
                                      raw_data.accelerometer.z,
                                      raw_data.gyroscope.x,
                                      raw_data.gyroscope.y,
                                      raw_data.gyroscope.z,
                                      raw_data.compass,
                                      ], dtype=np.float64)
            self.initialized = True

        while self.queue.qsize() > 0:
            raw_data = self.queue.get()
            self.imu_data = np.array([raw_data.accelerometer.x,
                                      raw_data.accelerometer.y,
                                      raw_data.accelerometer.z,
                                      raw_data.gyroscope.x,
                                      raw_data.gyroscope.y,
                                      raw_data.gyroscope.z,
                                      raw_data.compass,
                                      ], dtype=np.float64)

        return self.imu_data

    def __del__(self):
        self.imu.destroy()

        with self.queue.mutex:
            self.queue.queue.clear()


class CarlaEnv(gym.Env):
    def __init__(self, ep_length, route_file, env_id=0, train=True, eval=False):
        super(CarlaEnv, self).__init__()
        if env_id < 5:
            port = 2000 + 2 * env_id
            host = '192.168.0.4'
        else:
            port = 2000 + 2 * (env_id - 5)
            host = '192.168.0.5'
        if eval:
            port = 2000
            host = 'localhost'

        self._client = carla.Client(host, port)
        self._client.set_timeout(30.0)

        set_sync_mode(self._client, False)

        self._town_name = 'Town01'
        self._world = self._client.load_world(self._town_name)
        self._map = self._world.get_map()

        self._blueprints = self._world.get_blueprint_library()

        self._tick = 0
        self._player = None
        self.road_options = list(RoadOption)
        # vehicle, sensor
        self._actor_dict = collections.defaultdict(list)
        self._cameras = dict()
        self._sensors = dict()

        self.action_space = spaces.Box(low=-10, high=10,
                                       shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(9, 144, 256), dtype=np.uint8)

        self.metrics_space = spaces.Box(low=-100, high=100,
                                        shape=(3 + len(self.road_options),), dtype=np.float32)

        self.trajectory = parse_routes_file(route_file)

        self._waypoint_planner = RoutePlanner(1e-5, 5e-4)
        self._command_planner = RoutePlanner(1e-4, 2.5e-4, 258)

        self.ep_length = ep_length
        self.collision_sensor = None
        self.lane_sensor = None
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        set_sync_mode(self._client, True)

        self.train = train
        self.eval = eval

        self._spawn_player()
        self._setup_sensors()
        self.debug = Plotter(259, gps=True)

    def _spawn_player(self):
        start_pose = self.get_start_position()
        vehicle_bp = np.random.choice(self._blueprints.filter(VEHICLE_NAME))
        vehicle_bp.set_attribute('role_name', 'hero')

        self._player = self._world.spawn_actor(vehicle_bp, start_pose)
        self._actor_dict['player'].append(self._player)

        lane_bp = self._world.get_blueprint_library().find('sensor.other.lane_invasion')
        lane_location = carla.Location(0, 0, 0)
        lane_rotation = carla.Rotation(0, 0, 0)
        lane_transform = carla.Transform(lane_location, lane_rotation)
        ego_lane = self._world.spawn_actor(
            lane_bp, lane_transform, attach_to=self._player, attachment_type=carla.AttachmentType.Rigid)
        ego_lane.listen(lambda lane: self.lane_callback(lane))
        self.lane_sensor = ego_lane

        col_bp = self._world.get_blueprint_library().find('sensor.other.collision')
        col_location = carla.Location(0, 0, 0)
        col_rotation = carla.Rotation(0, 0, 0)
        col_transform = carla.Transform(col_location, col_rotation)
        ego_col = self._world.spawn_actor(
            col_bp, col_transform, attach_to=self._player, attachment_type=carla.AttachmentType.Rigid)
        ego_col.listen(lambda colli: self.col_callback(colli))
        self.collision_sensor = ego_col

    def lane_callback(self, lane):
        self.lane_invasion = True
        print("Lane invasion detected:\n"+str(lane)+'\n')

    def col_callback(self, colli):
        self.collision = True
        print("Collision detected:\n"+str(colli)+'\n')

    def _setup_sensors(self):
        """
        Add sensors to _actor_dict to be cleaned up.
        """
        self._cameras['rgb'] = Camera(
            self._world, self._player, 256, 144, 90, 1.2, 0.0, 1.3, 0.0, 0.0)
        self._cameras['rgb_left'] = Camera(
            self._world, self._player, 256, 144, 90, 1.2, -0.25, 1.3, 0.0, -45.0)
        self._cameras['rgb_right'] = Camera(
            self._world, self._player, 256, 144, 90, 1.2, 0.25, 1.3, 0.0, 45.0)

        self._sensors['gnss'] = GNSS(self._world, self._player)
        self._sensors['imu'] = IMU(self._world, self._player)

    def get_start_position(self):
        route_completed = (self._command_planner.route_completed() or
                           len(self._command_planner.route) == 0)
        training = self.train and not self.eval
        random_start = training and (
            not self._player or (
                not route_completed and
                np.random.randint(10) == 0
            )
        )

        if random_start:
            print('random_start')

        reset_position = route_completed or \
            self.eval or \
            random_start

        if reset_position:
            start = 0
            if random_start:
                start = np.random.randint(len(self.trajectory) - 2)
            global_plan_gps, global_plan_world_coord = interpolate_trajectory(
                self._world, self.trajectory[start:])

            self._waypoint_planner.set_route(
                global_plan_gps, global_plan_world_coord)
            ds_ids = downsample_route(global_plan_world_coord, 50)
            global_plan_gps = [global_plan_gps[x] for x in ds_ids]
            global_plan_world_coord = [
                global_plan_world_coord[x] for x in ds_ids]
            self._command_planner.set_route(
                global_plan_gps, global_plan_world_coord)
            start_pose = global_plan_world_coord[0][0]
            start_pose.location.z += 0.5

        else:
            transform = self._player.get_transform()
            start_pose = self._waypoint_planner.route[0][2]
            start_pose.location.z = transform.location.z

        return start_pose

    def reset_player_position(self):
        start_pose = self.get_start_position()

        ticks = 4
        velocity = carla.Vector3D()

        for _ in range(ticks):
            self._world.tick()
            self._player.set_transform(start_pose)
            self._player.set_target_angular_velocity(velocity)
            self._player.set_target_velocity(velocity)

    def reset(self):
        self.reset_player_position()

        for x in self._actor_dict['camera']:
            x.get()

        self._time_start = time.time()
        self._tick = 0

        self.cur_length = 0
        self.lane_invasion = False
        self.collision = False
        self.route_completed = False
        self.episode_reward = 0
        self.last_route_metrics = self._waypoint_planner.route_completion()

        obs, metrics, _, _, _ = self.step(None)

        return obs, metrics

    def tick_scenario(self):
        spectator = self._world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                self._player.get_location() + carla.Location(z=50),
                carla.Rotation(pitch=-90)))

    def step(self, action):
        control = carla.VehicleControl()

        if action is not None:
            control.steer = float(action[0])
            control.throttle = float(action[1])

        control.brake = 0.0
        self._player.apply_control(control)

        self._world.tick()
        self.tick_scenario()
        self._tick += 1

        transform = self._player.get_transform()

        # Put here for speed (get() busy polls queue).
        result = {key: val.get() for key, val in self._cameras.items()}

        rgb = result['rgb']
        rgb = np.transpose(rgb, (2, 0, 1))
        rgb_left = result['rgb_left']
        rgb_left = np.transpose(rgb_left, (2, 0, 1))
        rgb_right = result['rgb_right']
        rgb_right = np.transpose(rgb_right, (2, 0, 1))

        result = {key: val.get() for key, val in self._sensors.items()}
        gps = result['gnss']
        compass = result['imu'][-1]

        _, _, _ = self._waypoint_planner.run_step(gps)
        far_node, road_option, _ = self._command_planner.run_step(gps)

        rotation_matrix = np.array([
            [np.cos(compass), -np.sin(compass)],
            [np.sin(compass), np.cos(compass)]
        ])

        target = rotation_matrix.T.dot(far_node - gps)
        # metrics = np.concatenate(([speed], target))
        self.debug.clear()
        origin = np.array([0, 0])
        self.debug.dot(origin, target, (255, 0, 0))
        self.debug.dot(origin, origin, (0, 0, 255))
        self.debug.show()

        target *= 1000
        velocity = self._player.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])

        road_option_metrics = [
            1.0 if road_option == _road_option else 0.0 for _road_option in self.road_options]
        metrics = np.array([target[0], target[1], speed, *road_option_metrics])
        obs = np.concatenate((rgb, rgb_left, rgb_right)) / 255

        self.cur_length += 1

        done = False

        info = {
            'x': transform.location.x,
            'y': transform.location.y,
            'yaw': transform.rotation.yaw,
            'speed': speed,
            'gps_x': gps[0],
            'gps_y': gps[1],
            'compass': compass,
        }
        reward = 0
        if self.lane_invasion:
            reward -= 2

        if self.collision:
            reward -= 5

        route_metrics = self._waypoint_planner.route_completion()
        if self.last_route_metrics != route_metrics:
            reward += route_metrics - self.last_route_metrics

        self.episode_reward += reward
        self.last_route_metrics = route_metrics

        self.route_completed = self._waypoint_planner.route_completed()
        if (self.cur_length >= self.ep_length - 1
            or self.route_completed
            or self.collision
                or self.lane_invasion):
            info['episode'] = {'r': self.episode_reward, 'l': self.cur_length}
            done = True

        self.info = info
        reward = np.array(reward)

        return obs, metrics, reward, done, info

    def close(self):
        set_sync_mode(self._client, False)
        pass
