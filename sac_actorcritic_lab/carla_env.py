import os
import subprocess
import time
import random
import gym
import pygame
from gym.utils import seeding
from pygame.locals import *

from carla_util.hud import HUD
from carla_util.wrappers import *


class CarlaEnv(gym.Env):

    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, host="localhost", port=2000, #self, host="127.0.0.1", port=6000,
                 viewer_res=(1280, 720), obs_res=(1280, 720),
                 reward_fn=None, encode_state_fn=None, 
                 synchronous=True, fps=30, action_smoothing=0.9,
                 start_carla=True):
       
        # Start CARLA from CARLA_ROOT
        self.carla_process = None

        # Initialize pygame for visualization
        pygame.init()
        pygame.font.init()
        width, height = viewer_res

        out_width, out_height = obs_res
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.synchronous = synchronous

        # Setup gym environment
        self.seed()
        self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) # steer, throttle
        #self.action_space = gym.spaces.Box(np.float32(np.array([-1, 0])), np.float32(np.array([1, 1])), dtype=np.float32)


        #braking implementation
        # self.action_space = gym.spaces.Box(np.float32(np.array([-1, 0, 0])), np.float32(np.array([1, 1, 1])), dtype=np.float32) #steer, throttle, brake 
        
        #steering only
        #self.action_space = gym.spaces.Box(np.float32([-1]), np.float32([1]), dtype=np.float32)


        print('action_space',self.action_space) 

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*obs_res, 3), dtype=np.float32)
        self.metadata["video.frames_per_second"] = self.fps = self.average_fps = fps
        self.spawn_point = 1
        self.action_smoothing = action_smoothing
        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn

        self.world = None
        try:

            self.client = carla.Client(host, port)
            self.client.set_timeout(60.0)


            self.world = World(self.client) #

            if self.synchronous:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                #settings.fixed_delta_seconds = 0.05
                self.world.apply_settings(settings)


            lap_start_wp = self.world.map.get_waypoint(self.world.map.get_spawn_points()[0].location) #
            # lap_start_wp = self.world.map.get_waypoint(self.world.map.get_spawn_points()[i].location) #
            lap_end_wp = self.world.map.get_waypoint(self.world.map.get_spawn_points()[100].location) #
            print('lap_start_wp is:',lap_start_wp)
            # print('lap_end_wp is:',lap_end_wp)
            print('number of spawn_points', len(self.world.map.get_spawn_points()))
            # lap_start_wp = random.choice(self.waypoints) #Used to be waypoint[0]
            spawn_transform = lap_start_wp.transform
            spawn_transform.location += carla.Location(z=1.0)
            spawn_transform.location -= carla.Location(y=7.0)
            time.sleep(4) # - not to detect collision when the car spawnes from sky

            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, spawn_transform,
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

            self.colsensor = CollisionSensor(self.world, self.vehicle, on_collision_fn=lambda e: self._on_collision(e)) #
            self.laneinvsensor = LaneInvasionSensor(self.world, self.vehicle, on_invasion_fn=lambda e: self._on_invasion(e)) #


            # Create hud
            # print('Create hud') #
            self.hud = HUD(width, height)
            self.hud.set_vehicle(self.vehicle)
            self.world.on_tick(self.hud.on_world_tick)

            # Create cameras
            # print('Create Camera') #
            self.dashcam = Camera(self.world, out_width, out_height,
                                  transform=camera_transforms["dashboard"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image(e),
                                  sensor_tick=0.0 if self.synchronous else 1.0/self.fps)
            self.camera  = Camera(self.world, width, height,
                                  transform=camera_transforms["spectator"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e),
                                  sensor_tick=0.0 if self.synchronous else 1.0/self.fps)

            self.semantic_snesor  = Camera(self.world, out_width, out_height,
                                  transform=camera_transforms["dashboard"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_semantic_segmentation_image(e),
                                  sensor_tick=0.0 if self.synchronous else 1.0/self.fps, camera_type="sensor.camera.semantic_segmentation", color_converter=carla.ColorConverter.CityScapesPalette)

            #
            self.actor_list = []
            self.actor_list.append(self.vehicle)
            self.actor_list.append(self.colsensor)
            self.actor_list.append(self.laneinvsensor)
            self.actor_list.append(self.dashcam)
            self.actor_list.append(self.camera)
            self.actor_list.append(self.semantic_snesor)

        except Exception as e:
            self.close()
            raise e

        # Reset env to set initial state
        self.reset()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, is_training=True):

        self.collision_hist = []
        self.laneinvasion_hist = []
        self.obstacle_data=[] 
        self.actor_list = []
        # Do a soft reset (teleport vehicle)
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)

        self.vehicle.tick()
        lap_start_wp = self.world.map.get_waypoint(self.world.map.get_spawn_points()[0].location) #
        spawn_transform = lap_start_wp.transform #
        spawn_transform.location += carla.Location(z=1.0) #
        spawn_transform.location -= carla.Location(y=7.0) #
        self.vehicle.set_transform(spawn_transform) #

        # Give 2 seconds to reset
        print('give 2 seconds to reset')  #
        if self.synchronous:
            ticks = 0
            while ticks < self.fps * 2:
                self.world.tick()
                #print('world.tick')  #
                try:
                    ticks += 3  #
                    self.world.wait_for_tick(seconds=1.0/self.fps + 0.1)
                except:
                    pass
        else:
            time.sleep(2.0)
        # print('tick done')  #
        self.terminal_state = False 
        self.closed = False         
        self.extra_info = []        
        self.observation = self.observation_buffer = None   
        self.viewer_image = self.viewer_image_buffer = None 
        self.semantic_segmentation = self.semantic_segmentation_buffer = None   
        self.start_t = time.time()
        self.step_count = 0
        self.is_training = is_training

        # Metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.episode_start_location = self.vehicle.get_transform().location #
        self.previous_rotation = self.vehicle.get_transform().rotation #
        self.distance_traveled = 0.0
        self.distance_from_start = 0.0
        self.angle_difference = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.laps_completed = 0.0

        # Return initial observation
        return self.step(None)[0]

    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
        pygame.quit()
        if self.world is not None: #
            self.world.destroy() #
        self.closed = True
        # print('def close end')  #

    # def render(self, mode="human"):
    def render(self, mode="state_pixels"):

        # Build image from spectator camera
        # self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Render to screen
        # pygame.display.flip() 

        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), (100, 100))
            pygame.display.flip()   # Render to screen
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation

    def step(self, action):
        self.laneinvasion_hist = []
        # print('def step start')
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")

        # Asynchronous update logic
        if not self.synchronous:
            if self.fps <= 0:
                # Go as fast as possible
                self.clock.tick()
            else:
                # Sleep to keep a steady fps
                self.clock.tick_busy_loop(self.fps)
            
            # Update average fps (for saving recordings)
            if action is not None:
                self.average_fps = self.average_fps * 0.5 + self.clock.get_fps() * 0.5

        # Take action
        if action is not None:
            steer, throttle = [float(a) for a in action]

            #Adding Braking
            # steer, throttle, brake = [float(a) for a in action]
            # steer = [float(a) for a in action]
            # steer = steer * 0.7
            # throttle = abs(throttle) + 0.2
            #brake = max(brake - 0.5, 0.0)
            
            # self.vehicle.control.steer = self.vehicle.control.steer * self.action_smoothing + action[0] * (1.0 - self.action_smoothing)
            # self.vehicle.control.throttle = 0.65

            self.vehicle.control.steer    = self.vehicle.control.steer * self.action_smoothing + steer * (1.0-self.action_smoothing)
            self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (1.0-self.action_smoothing)
            #self.vehicle.control.brake = self.vehicle.control.brake * self.action_smoothing + brake * (1.0-self.action_smoothing)

            # print(self.vehicle.get_speed())
            
            print('step {}: {}'.format(self.step_count, [round(steer, 3), round(throttle, 3)]))
            # print('step {}: {}'.format(self.step_count, [round(steer, 3), round(throttle, 3), round(brake, 3)]))
            # print('step {}: {}'.format(self.step_count, round(self.vehicle.control.steer, 3)))

        # Tick game
        self.hud.tick(self.world, self.clock)
        self.world.tick()


        if self.synchronous:
            self.world.tick()

        # Get most recent observation and viewer image
        self.viewer_image = self._get_viewer_image()
        self.observation = self._get_observation()
        self.sem_image = self._get_semantic_segmentation_image() #
        self.encoded_state = self.encode_state_fn(self)
        # print('observation shape',self.observation.shape)

        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Calculate distance traveled
        self.distance_traveled += self.previous_location.distance(transform.location)  #
        self.previous_location = transform.location #

        self.distance_from_start = self.episode_start_location.distance(transform.location)  # distance to the start loc
        
        self.angle_difference += self.previous_rotation.yaw - transform.rotation.yaw #

        self.previous_rotation = transform.rotation #


        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()
        
        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward
        self.step_count += 1

        # Check for ESC press
        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            self.close()
            self.terminal_state = True
        
        return self.encoded_state, self.last_reward, self.terminal_state, { "closed": self.closed }
        # print('end of def step')  #

    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image


    def _get_semantic_segmentation_image(self):
        while self.semantic_segmentation_buffer is None:
            pass
        sem_image = self.semantic_segmentation_buffer.copy()
        # processed_sem_image = self.process_img(sem_image) 
        self.semantic_segmentation_buffer = None
        return sem_image


    def _on_collision(self, event):
        self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))
        self.collision_hist.append(event) #

    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))
        self.laneinvasion_hist.append(text) #

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image


    def _set_semantic_segmentation_image(self, image):
        self.semantic_segmentation_buffer = image  


def reward_fn(env):
    early_termination = False
    if early_termination:
        if time.time() - env.start_t > 3.0 and env.vehicle.get_speed() < 1.0 / 3.6: # changed 5s to 3s
            env.terminal_state = True

        #
        if len(env.collision_hist)!=0:
            env.terminal_state = True
    return 0

if __name__ == "__main__":
    # Example of using CarlaEnv with keyboard controls
    env = CarlaEnv(obs_res=(160, 80), reward_fn=reward_fn)
    action = np.zeros(env.action_space.shape[0])
    while True:
        env.reset(is_training=True)
        while True:
            # Process key inputs
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[K_LEFT] or keys[K_a]:
                action[0] = -0.5
            elif keys[K_RIGHT] or keys[K_d]:
                action[0] = 0.5
            else:
                action[0] = 0.0
            action[0] = np.clip(action[0], -1, 1)

            
            action[1] = 1.0 if keys[K_UP] or keys[K_w] else 0.0

            # # Braking Implementation
            # action[2] = 1.0 if keys[K_SPACE] else 0.0

            # Take action
            obs, _, done, info = env.step(action)
            if info["closed"]: # Check if closed
                exit(0)
            env.render() # Render
            if done: break
    env.close()
