import numpy as np
import carla
import math


low_speed_timer = 0
max_diff = 2.0


def create_reward_fn(reward_fn, max_speed=-1):

    def func(env):
        global low_speed_timer
        global max_diff

        terminal_reason = "Running..."

        low_speed_timer += 1.0 / env.fps
        speed = env.vehicle.get_speed()
        
        if low_speed_timer > 3.0 and speed < 1.0 / 3.6:
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        if abs(env.angle_difference) > np.deg2rad(180):
            env.terminal_state = True
            terminal_reason = "Vehicle circle"

        if max_speed > 0 and speed > max_speed:
            env.terminal_state = True
            terminal_reason = "Too fast"

        if len(env.collision_hist) != 0:
            env.terminal_state = True
            terminal_reason = "Collision"

        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        elif len(env.collision_hist) == 0:  # collision reward
            low_speed_timer = 0.0
            reward -= 10
        else:
            reward -= 1

        if env.terminal_state:
            print("Terminated due to:", terminal_reason)
            env.extra_info.extend([terminal_reason, ""])
        
        return reward
    return func


def scale_reward(value, max_value, scale_factor):
    return max(1.0 - (1.0 / max_value) * abs(value), 0) * scale_factor


def reward_old(env):
    min_speed = 15.0  # km/h
    max_speed = 35.0  # km/h
    target_speed = (min_speed + max_speed) / 2

    speed_kmh = 3.6 * env.vehicle.get_speed()
    speed_reward = scale_reward(speed_kmh, min_speed, 1.0)
    
    center_diff = env.world.map.get_waypoint(env.vehicle.get_location()).transform.location.x - env.vehicle.get_location().x
    centering_reward = scale_reward(center_diff, max_diff, 1.0)

    angle_diff = abs(math.radians(env.world.map.get_waypoint(env.vehicle.get_location()).transform.rotation.yaw) - math.radians(env.vehicle.get_transform().rotation.yaw))
    angle_deviation_reward = scale_reward(angle_diff, np.deg2rad(20), 1.0)

    reward = speed_reward + centering_reward + angle_deviation_reward
    return reward

reward_functions = {}

reward_functions["reward_old"] = create_reward_fn(reward_old)


def reward_final(env):
    min_speed = 15.0  # km/h
    max_speed = 35.0  # km/h
    
    target_speed = (min_speed + max_speed) / 2

    speed_kmh = 3.6 * env.vehicle.get_speed()
    speed_reward = scale_reward(speed_kmh, min_speed, 1.0)
    
    center_diff = env.world.map.get_waypoint(env.vehicle.get_location()).transform.location.x - env.vehicle.get_location().x
    centering_reward = scale_reward(center_diff, max_diff, 1.0)

    angle_diff = abs(math.radians(env.world.map.get_waypoint(env.vehicle.get_location()).transform.rotation.yaw) - math.radians(env.vehicle.get_transform().rotation.yaw))
    angle_deviation_reward = scale_reward(angle_diff, np.deg2rad(20), 1.0)

    lane_width = 4.0 
    lane_deviation = abs(center_diff) - lane_width / 2
    deviation_penalty = -0.5 * max(lane_deviation, 0.0)

    if len(env.laneinvasion_hist) == 0:
        lane_pen = 0
    elif "'SolidSolid'" in env.laneinvasion_hist[-1]:
        lane_pen = -3 
    elif "'Solid'" in env.laneinvasion_hist[-1]:
        lane_pen = -2 * len(env.laneinvasion_hist)
    else:
        lane_pen = -0.1 

    reward = (
        speed_reward *
        centering_reward *
        angle_deviation_reward + 
        lane_pen + 
        deviation_penalty
    )
    return reward

reward_functions["reward_final"] = create_reward_fn(reward_final)
