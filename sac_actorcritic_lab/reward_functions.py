import numpy as np
import carla
import math


low_speed_timer = 0
target_speed    = 20.0 # kmh
max_diff = 2.0


def create_reward_fn(reward_fn, max_speed=-1):

    def func(env):
        terminal_reason = "Running..."

        # Stop if speed is less than 1.0 km/h after the first 5s of an episode
        global low_speed_timer
        global max_diff
        
        low_speed_timer += 1.0 / env.fps
        speed = env.vehicle.get_speed()
        if low_speed_timer > 3.0 and speed < 1.0 / 3.6: # changed 5s to 3s
            env.terminal_state = True
            terminal_reason = "Vehicle stopped"

        if abs(env.angle_difference) > 1800: # 
            env.terminal_state = True
            terminal_reason = "Vehicle circle"

        # Stop if speed is too high
        if max_speed > 0 and speed_kmh > max_speed:
            env.terminal_state = True
            terminal_reason = "Too fast"

        if len(env.collision_hist)!=0:
            env.terminal_state = True
            terminal_reason = "Collision"

        # Calculate reward
        reward = 0
        if not env.terminal_state:
            reward += reward_fn(env)
        elif len(env.collision_hist)==0:  # collision reward
            low_speed_timer = 0.0
            reward -= 10   #if car doesn't move
        else:
            reward -= 1

        if env.terminal_state:
            print("Terminated due to: ", terminal_reason)
            env.extra_info.extend([
                terminal_reason,
                ""
            ])
        return reward
    return func

# Create reward functions dict
reward_functions = {}

def reward_old(env):
    global max_diff
    min_speed = 15.0  # km/h
    max_speed = 35.0  # km/h
    max_distance = 3.0
    
    target_speed = (min_speed + max_speed) / 2

    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh < min_speed:
        speed_reward = speed_kmh / min_speed
    elif speed_kmh > target_speed:
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:
        speed_reward = 1.0

    ### Center Deviation Reward 
    center_diff =  env.world.map.get_waypoint(env.vehicle.get_location()).transform.location.x - env.vehicle.get_location().x
    centering_reward = max(1.0 - (1.0 / max_diff) * abs(center_diff), 0) #scaling

    ### Angle Deviation Reward
    angle_diff = abs(math.radians(env.world.map.get_waypoint(env.vehicle.get_location()).transform.rotation.yaw) - math.radians(env.vehicle.get_transform().rotation.yaw))
    angle_deviation_reward = max(1.0 - (1.0 / np.deg2rad(20)) * angle_diff, 0) #scaling


    reward = (
        speed_reward +
        centering_reward +
        angle_deviation_reward
    )
    return reward

reward_functions["rewawrd_old"] = create_reward_fn(reward_old)


def reward_final(env):
    global max_diff

    ### Speed Reward
    min_speed = 15.0  # km/h
    max_speed = 35.0  # km/h
    
    target_speed = (min_speed + max_speed) / 2

    speed_kmh = 3.6 * env.vehicle.get_speed()
    if speed_kmh < min_speed:
        speed_reward = speed_kmh / min_speed
    elif speed_kmh > target_speed:
        speed_reward = 1.0 - (speed_kmh - target_speed) / (max_speed - target_speed)
    else:
        speed_reward = 1.0
    
    ### Center Deviation Reward 
    center_diff =  env.world.map.get_waypoint(env.vehicle.get_location()).transform.location.x - env.vehicle.get_location().x
    centering_reward = max(1.0 - (1.0 / max_diff) * abs(center_diff), 0) #scaling

    ### Angle Deviation Reward
    angle_diff = abs(math.radians(env.world.map.get_waypoint(env.vehicle.get_location()).transform.rotation.yaw) - math.radians(env.vehicle.get_transform().rotation.yaw))
    angle_deviation_reward = max(1.0 - (1.0 / np.deg2rad(20)) * angle_diff, 0) #scaling

    ### Lane Deviation Penalty
    lane_width = 4.0 
    lane_deviation = abs(center_diff) - lane_width / 2
    if lane_deviation > 0:
        deviation_penalty = -0.5 * lane_deviation #scaling
    else:
        deviation_penalty = 0.0

    ### Lane Invasion Penalty
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
