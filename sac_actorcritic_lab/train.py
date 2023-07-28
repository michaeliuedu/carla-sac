import argparse
import os
import numpy as np
import cv2
from torch import nn
from ReplayMemory import ReplayMemory
from torch_agent import TorchAgent, TorchModel
from torch_sac import TorchSAC
from reward_functions import reward_functions
from carla_env import CarlaEnv
from torch.utils.tensorboard import SummaryWriter

import torch
# import carla


# Defining the model
class VAE(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self, in_channels: int=3, img_width = 160, img_height = 96, z_dim: int = 64):
        super().__init__()

        hid_dims = [32, 64, 128, 256, 512]  ##hidden dimensions
        self.LAST_HID_DIM = hid_dims[-1]
        self.z_dim = z_dim
        if self.LAST_HID_DIM == 512:
            W_RATIO = int(img_width/32)
            H_RATIO = int(img_height/32)
        elif self.LAST_HID_DIM == 256:
            W_RATIO = int(img_width/16)
            H_RATIO = int(img_height/16)

        modules = []

        ##building the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels = hid_dims[0], kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(hid_dims[0]), nn.LeakyReLU(),
            nn.Conv2d(hid_dims[0], out_channels = hid_dims[1], kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(hid_dims[1]), nn.LeakyReLU(),
            nn.Conv2d(hid_dims[1], out_channels = hid_dims[2], kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(hid_dims[2]), nn.LeakyReLU(),
            nn.Conv2d(hid_dims[2], out_channels = hid_dims[3], kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(hid_dims[3]), nn.LeakyReLU(),
            nn.Conv2d(hid_dims[3], out_channels = hid_dims[4], kernel_size = 3, stride = 2, padding = 1), nn.BatchNorm2d(hid_dims[4]), nn.LeakyReLU(),
        )
        # for h_dim in hid_dims:
        #     modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=h_dim, kernel_size= 3, stride= 2, padding  = 1), nn.BatchNorm2d(h_dim), nn.LeakyReLU()))
        #     in_channels = h_dim

        # self.encoder = nn.Sequential(*modules)

        self.fc_mu = nn.Linear(hid_dims[-1] * W_RATIO * H_RATIO, self.z_dim)  ##if 
        self.fc_var = nn.Linear(hid_dims[-1] * W_RATIO * H_RATIO, self.z_dim)  

        #building the decoder
        modules = []
        self.decoder_input = nn.Linear(self.z_dim, hid_dims[-1] * W_RATIO * H_RATIO)
        hid_dims.reverse() ##reversing the hidden dimensions

        # for i in range(len(hid_dims) - 1):
        #     modules.append(nn.Sequential(nn.ConvTranspose2d(hid_dims[i], hid_dims[i + 1], kernel_size=3, stride = 2, padding=1, output_padding=1), nn.BatchNorm2d(hid_dims[i + 1]), nn.LeakyReLU()))

        # self.decoder = nn.Sequential(*modules, nn.ConvTranspose2d(hid_dims[-1], hid_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hid_dims[-1]), nn.LeakyReLU(),                            
        #                     nn.Conv2d(hid_dims[-1], out_channels= 3, kernel_size= 3, padding= 1), nn.Sigmoid())

        # self.final_layer = nn.Sequential(nn.ConvTranspose2d(hid_dims[-1], hid_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.BatchNorm2d(hid_dims[-1]), nn.LeakyReLU(),                            
        #                     nn.Conv2d(hid_dims[-1], out_channels= 3, kernel_size= 3, padding= 1), nn.Sigmoid())
                                      
                            
        #building the decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hid_dims[0], hid_dims[1], kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(hid_dims[1]), nn.LeakyReLU(),

            nn.ConvTranspose2d(hid_dims[1], hid_dims[2], kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(hid_dims[2]), nn.LeakyReLU(),

            nn.ConvTranspose2d(hid_dims[2], hid_dims[3], kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(hid_dims[3]), nn.LeakyReLU(),
                
            nn.ConvTranspose2d(hid_dims[3], hid_dims[4], kernel_size=3, stride = 2, padding=1, output_padding=1),
                nn.BatchNorm2d(hid_dims[4]), nn.LeakyReLU(),

            nn.ConvTranspose2d(hid_dims[-1], hid_dims[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hid_dims[-1]), nn.LeakyReLU(),
            
            nn.Conv2d(hid_dims[-1] , out_channels= 3, kernel_size= 3, padding= 1), nn.Sigmoid()  ##final layer   ##or nn.Tanh() 

            )


    def encode(self, input):
 
        result = self.encoder(input)
        # print('encode result shape', result.shape)
        result = torch.flatten(result, start_dim=1)
        # print('encode result shape after flattening', result.shape)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        # print('decoder input shape', result.shape)
        result = result.view(-1, self.LAST_HID_DIM, H_RATIO, W_RATIO)
        result = self.decoder(result)
        # print('decoder output shape', result.shape)
        return result

    def reparameterize(self, mu, logvar):

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = eps * std + mu
            # print('reparameterize output shape',z.shape)
            return z
        else:
            return mu

    def forward(self, input):
        mu, slog_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        # print('z shape',z.shape)
        y_hat_rgb = self.decode(z)
        return y_hat_rgb, mu, log_var


    def sample(self, num_samples): 

        z = torch.randn(num_samples, self.z_dim)
        # z = z.to(device)
        samples = self.decode(z)
        return samples
    

    def generate(self, x):
        return self.forward(x)[0]
        
class VideoRecorder():
    def __init__(self, filename, frame_size, fps=30):
        self.video_writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"MPEG"), int(fps),
            (frame_size[1], frame_size[0]))

    def add_frame(self, frame):
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()

    def __del__(self):
        self.release()


def vector(v):
    """ Turn carla Location/Vector3D/Rotation to np.array """
    if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
        return np.array([v.x, v.y, v.z])
    elif isinstance(v, carla.Rotation):
        return np.array([v.pitch, v.yaw, v.roll])


def preprocess_frame(frame):
    frame = frame.astype(np.float32) / 255.0
    return frame

def create_encode_state_fn( pretrained_model, measurements_to_include):


    # Turn into bool array for performance
    measure_flags = ["steer" in measurements_to_include,
                     "throttle" in measurements_to_include,
                     "speed" in measurements_to_include,
                     "orientation" in measurements_to_include]

    def encode_state(env):


        frame = preprocess_frame(env.observation[26:-6, :, :])
        frame = torch.tensor(np.array([frame.transpose(2, 0, 1)]), dtype=torch.float)
        encoded_state = pretrained_model.encode(frame)[0].detach().numpy() #encode state
        
        # Append measurements
        measurements = []
        if measure_flags[0]: measurements.append(env.vehicle.control.steer)
        if measure_flags[1]: measurements.append(env.vehicle.control.throttle)
        if measure_flags[2]: measurements.append(env.vehicle.get_speed())

        # Orientation could be usedful for predicting movements that occur due to gravity
        if measure_flags[3]: measurements.extend(vector(env.vehicle.get_forward_vector()))

        # encoded_state = np.append(encoded_state, measurements)
        return encoded_state
    return encode_state


def main():


    # Read parameters
    WARMUP_STEPS = 2e3
    EVAL_EPISODES = 2
    MEMORY_SIZE = int(1e7)
    BATCH_SIZE = 1024
    GAMMA = 0.99
    TAU = 0.005
    ALPHA =  0.01 #0.2  # determines the relative importance of entropy term against the reward
    ACTOR_LR = 3e-4
    CRITIC_LR = 3e-4
    vae_z_dim        = 64  #128
    synchronous      = True
    fps              = 30
    action_smoothing = 0.0
    reward_fn        = "reward_final"
    seed             = 0
    eval_interval    = 5
    # record_eval      = True
    IMG_W            = 160   
    IMG_H            = 128  #
    model_name       = 'ali_' 
    start_carla      = True
    n_episodes       = 3000


    USE_PRETRAINED = True
    

    pretrained_model = VAE(z_dim = vae_z_dim)
    pretrained_model.load_state_dict(torch.load('model_seg_160_96.pt'))
    pretrained_model.eval()

   

    # Create state encoding fn
    #measurements_to_include = set(["steer", "throttle", "speed"])
    measurements_to_include = set([])


    print("")
    print("Training parameters:")

    # input_shape = np.array([vae_z_dim + len(measurements_to_include)]) #64 + 3 = 67
    input_shape = np.array([vae_z_dim])

    encode_state_fn = create_encode_state_fn(pretrained_model, measurements_to_include)


    # Create env
    print("Creating environment")
    print( 'action_smoothing=',action_smoothing,
                   'encode_state_fn=',encode_state_fn,
                   'reward_fn=',reward_functions[reward_fn],
                   'synchronous=',synchronous,
                   'fps=',fps,
                   'input dimentions=',input_shape)
    env = CarlaEnv(obs_res=(IMG_W, IMG_H),
                   action_smoothing=action_smoothing,
                   encode_state_fn=encode_state_fn, 
                   reward_fn=reward_functions[reward_fn],
                   synchronous=synchronous,
                   fps=fps,
                   start_carla=start_carla)
    print('Environment created')
    if isinstance(seed, int):
        env.seed(seed)
    # best_eval_reward = -float("inf")

    # Environment constants
    num_actions = env.action_space.shape[0]


    env_num = 1

    eval_env = env

    action_dim = env.action_space.shape[0]

    # Initialize model, algorithm, agent, replay_memory
    CarlaModel, SAC, CarlaAgent = TorchModel, TorchSAC, TorchAgent

    model = CarlaModel(input_shape, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR)
    agent = CarlaAgent(algorithm)

    
    if USE_PRETRAINED:
        agent.restore('./{}_model/old/step_{}_model.ckpt'.format(args.framework, 5916))
        print("Loaded pretrained model: {}".format(agent))
    


    rpm = ReplayMemory(max_size=MEMORY_SIZE, obs_dim=input_shape[0], act_dim=action_dim)   

    total_steps = 0
    last_save_steps = 0
    test_flag = 0
    video_flag = 0
    best_eval_reward = -float("inf")

    # obs_list = env.reset()
    state, _, _ = env.reset(), False, 0

    tb = SummaryWriter()
    # For every episode
    for episode in range(n_episodes):
        
        rewards_history = []
        avg_rewards = -400

        ####Evaluation Function copied here: Run evaluation periodically
        if episode % eval_interval == 0:
            video_filename = os.path.join("videos_throttle/", "episode{}.mp4".format(episode))
            
            if USE_PRETRAINED:
                video_filename = os.path.join("videos/pretrained", "episode{}.mp4".format(episode))

            # Init test env
            state, terminal_state, total_reward = env.reset(is_training=False), False, 0
            rendered_frame = env.render(mode="rgb_array")

            # Init video recording
            if video_filename is not None:
                print("Recording video to {} ( {}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape, int(env.average_fps)))
                video_recorder = VideoRecorder(video_filename,
                                               frame_size=rendered_frame.shape,
                                               fps=env.average_fps)
                video_recorder.add_frame(rendered_frame)
            else:
                video_recorder = None

            while not terminal_state:

                env.extra_info.append("Episode {}".format(episode))
                env.extra_info.append("Running eval...".format(episode))
                env.extra_info.append("")

                env.collision_hist = []
                # Train episode
                if rpm.size() < WARMUP_STEPS:
                    action_list = np.random.uniform(-1, 1, size=action_dim)   #for _ in range(env_num)]


                else:
                   
                    action_list = agent.sample(state) #generates sample actions (list of throttle, steer) based on state (VAE encoded image data and prior car measurements)

                #observes the next action to take based on such actions
                new_state, reward, terminal_state, info = env.step(action_list)

                rpm.append(state, action_list, reward, new_state, terminal_state) ##when we have only one env (env_num=1) 
                # state = env.get_obs()
                state = env.encoded_state #encode state
                total_steps = env.step_count
                # Train agent after collecting sufficient data
                if rpm.size() >= WARMUP_STEPS:
                    batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(BATCH_SIZE)
                    #update agent based on such rewards and generated outputs
                    agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal)

                if info["closed"] == True:
                    break
                # Add frame
                rendered_frame = env.render(mode="rgb_array")
                if video_recorder is not None:
                    video_recorder.add_frame(rendered_frame)
                total_reward += reward

            # Release video
            if video_recorder is not None:
                video_recorder.release()
            
            if info["closed"] == True:
                exit(0)

            eval_reward = total_reward

            if eval_reward > best_eval_reward:
                # model.save()
                agent.save('./{}_model/step_{}_model.ckpt'.format(args.framework, total_steps))
                # last_save_steps = total_steps
                best_eval_reward = eval_reward

        # Reset environment
        state, terminal_state, total_reward = env.reset(), False, 0

        print(f"Episode {episode} (Step {env.step_count})")

        # while total_steps < args.train_total_steps:
        while not terminal_state:
                        
            env.collision_hist = []
            # Train episode
            if rpm.size() < WARMUP_STEPS:
                action_list = np.random.uniform(-1, 1, size=action_dim)   #for _ in range(env_num)]

            else:
                action_list = agent.sample(state) 


            new_state, reward, terminal_state, info = env.step(action_list)

            rpm.append(state, action_list, reward, new_state, terminal_state) ##when we have only one env (env_num=1) 
            # state = env.get_obs()
            state = env.encoded_state
            total_steps = env.step_count
            # Train agent after collecting sufficient data
            if rpm.size() >= WARMUP_STEPS:
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(BATCH_SIZE)
                agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal)
            
            if terminal_state:
                break

            if info["closed"] == True:
                exit(0)

            cv2.imshow("Actor-critic agent_rgb - train",env.observation[26:-6, :, :])
            cv2.waitKey(1)
            total_reward += reward

            rewards_history.append(total_reward)
            avg_rewards = np.mean(rewards_history[-100:])

        tb.add_scalar("average_reward", avg_rewards, episode)
        tb.add_scalar("total reward", total_reward, episode)
        tb.add_scalar("distance_moved", env.distance_traveled, episode)
        tb.add_scalar("distance_traveled", env.distance_from_start, episode)
        tb.add_scalar("roration_turned", env.angle_difference, episode)
        tb.add_scalar("average_speed", 3.6 * env.speed_accum / env.step_count, episode)
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xparl_addr",
        default='localhost:8080',
        help='xparl address for parallel training')
    parser.add_argument("--env", default="carla-v0")
    parser.add_argument(
        '--framework',
        default='paddle',
        help='choose deep learning framework: torch or paddle')
    parser.add_argument(
        "--train_total_steps",
        default=5e5,
        type=int,
        help='max time steps to run environment')
    parser.add_argument(
        "--test_every_steps",
        default=1e3,
        type=int,
        help='the step interval between two consecutive evaluations')
    parser.add_argument(
        "--video_record_steps",
        default = 2 * 1e3,
        type=int,
        help='the step interval between two consecutive video recordings')
    args = parser.parse_args()

    main()