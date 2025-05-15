# solves LunarLander-v3 using DQN

import gymnasium
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import time
import multiprocessing as mp
import threading
import queue
import datetime
import os
import imageio
from copy import deepcopy
import traceback

CKPT_FREQ = 100_000

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def worker(worker_id, env_name, input_dim, output_dim, transition_queue, param_recv_conn, stop_event):
    try:
        env = gymnasium.make(env_name, render_mode="rgb_array")
        local_q_net = QNetwork(input_dim, output_dim).to('cpu')
        state_dict, local_epsilon = param_recv_conn.recv()
        local_q_net.load_state_dict(state_dict)
        
        obs, info = env.reset()
        
        while not stop_event.is_set():
            if np.random.rand() < local_epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32)
                    q_values = local_q_net(obs_tensor)
                    action = q_values.argmax().item()
            
            next_obs, rewards, done, truncated, info = env.step(action)
            
            transition = (obs.flatten(), action, rewards, next_obs.flatten(), done or truncated)
            try:
                transition_queue.put((worker_id, transition), timeout=0.2)
            except queue.Full:
                if stop_event.is_set():
                    break
                time.sleep(0.2)
                continue
            
            if done or truncated:
                obs, info = env.reset()
            else:
                obs = next_obs
            
            if param_recv_conn.poll(0):
                try:
                    state_dict, local_epsilon = param_recv_conn.recv()
                    local_q_net.load_state_dict(state_dict)
                except EOFError:
                    break
    
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        traceback.print_exception(type(e), e, e.__traceback__)
    
    param_recv_conn.close()

class DQN:
    def __init__(self, env_name, learning_rate=5e-4, buffer_size=15000, batch_size=256, gamma=0.8, 
             exploration_steps=1e5, exploration_final_eps=0.05, verbose=1, 
             tensorboard_log=None, num_workers=16, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.exploration_steps = exploration_steps
        self.exploration_final_eps = exploration_final_eps
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log
        self.num_workers = num_workers
        self.device = device

        with gymnasium.make(env_name, render_mode="rgb_array") as temp_env:
            input_dim = np.prod(temp_env.observation_space.shape)
            output_dim = temp_env.action_space.n

        self.q_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net = QNetwork(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.replay_buffer = deque(maxlen=buffer_size)

        if tensorboard_log:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(tensorboard_log, f"run_{timestamp}")
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        self.total_timesteps = None
        self.train_step = 0
        self.log_dir = log_dir

    def get_epsilon(self, step):
        if step < self.exploration_steps:
            return 1.0 - (step / self.exploration_steps) * (1.0 - self.exploration_final_eps)
        return self.exploration_final_eps

    def sample(self, batch_size):
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            max_q_next = self.target_net(next_states).max(1)[0]
            targets = rewards + self.gamma * (1 - dones) * max_q_next
        q_values = self.q_net(states).gather(1, actions.view(-1, 1)).view(-1)
        loss = ((q_values - targets) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.writer:
            self.writer.add_scalar("Loss/train", loss.item(), self.train_step)
            self.train_step += 1

    def training_loop(self, stop_event):
        while not stop_event.is_set():
            if len(self.replay_buffer) >= self.batch_size:
                self.train()
            time.sleep(0.01)

    def learn(self, total_timesteps):
        self.total_timesteps = total_timesteps
        total_steps = 0
        
        transition_queue = mp.Queue(maxsize=1000)
        param_pipes = [mp.Pipe(duplex=True) for _ in range(self.num_workers)]
        main_send_conns = [pipe[0] for pipe in param_pipes]
        worker_recv_conns = [pipe[1] for pipe in param_pipes]
        stop_event = mp.Event()
        
        with gymnasium.make(self.env_name, render_mode="rgb_array") as temp_env:
            input_dim = np.prod(temp_env.observation_space.shape)
            output_dim = temp_env.action_space.n
        workers = [mp.Process(target=worker, args=(i, self.env_name, input_dim, output_dim, transition_queue, worker_recv_conns[i], stop_event)) 
                   for i in range(self.num_workers)]
        
        current_episode_rewards = [0.0] * self.num_workers
        episode_rewards = deque(maxlen=100)
        
        try:
            for w in workers:
                w.daemon = True
                w.start()
            
            initial_state_dict = {k: v.cpu() for k, v in self.q_net.state_dict().items()}
            initial_epsilon = self.get_epsilon(0)
            for conn in main_send_conns:
                conn.send((initial_state_dict, initial_epsilon))
            
            training_thread = threading.Thread(target=self.training_loop, args=(stop_event,))
            training_thread.daemon = True
            training_thread.start()
            
            while total_steps < total_timesteps:
                try:
                    worker_id, transition = transition_queue.get(timeout=1.0)
                    state, action, reward, next_state, episode_done = transition
                    self.replay_buffer.append(transition)
                    
                    current_episode_rewards[worker_id] += reward
                    if episode_done:
                        episode_rewards.append(current_episode_rewards[worker_id])
                        current_episode_rewards[worker_id] = 0.0
                        if self.writer:
                            self.writer.add_scalar("Episode Reward", episode_rewards[-1], total_steps)
                    
                    total_steps += 1
                    
                    if total_steps % CKPT_FREQ == 0:
                        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{total_steps}.pth")
                        self.save(checkpoint_path)
                        if self.verbose > 0:
                            print(f"Saved checkpoint at step {total_steps} to {checkpoint_path}")
                        
                        video_dir = os.path.join(self.log_dir, "videos")
                        os.makedirs(video_dir, exist_ok=True)
                        video_path = os.path.join(video_dir, f"video_step_{total_steps}.mp4")
                        
                        # Create a CPU copy of the q_net for inference
                        q_net_cpu = deepcopy(self.q_net).to("cpu")
                        q_net_cpu.eval()
                        
                        with gymnasium.make(self.env_name, render_mode="rgb_array") as test_env:
                            obs, info = test_env.reset()
                            frames = []
                            done = False
                            while not done:
                                with torch.no_grad():
                                    obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32)
                                    q_values = q_net_cpu(obs_tensor)
                                    action = q_values.argmax().item()
                                next_obs, rewards, done, truncated, info = test_env.step(action)
                                frame = test_env.render()
                                frames.append(frame)
                                obs = next_obs
                                done = done or truncated
                            
                            with imageio.get_writer(video_path, fps=30) as video_writer:
                                for frame in frames:
                                    video_writer.append_data(frame)
                            
                            if self.verbose > 0:
                                print(f"Saved test video at step {total_steps} to {video_path}")
                    
                    if total_steps % 1000 == 0:
                        # send the current state_dict and epsilon to workers
                        state_dict_cpu = {k: v.cpu() for k, v in self.q_net.state_dict().items()}
                        epsilon = self.get_epsilon(total_steps)
                        for conn in main_send_conns:
                            conn.send((state_dict_cpu, epsilon))
                        self.target_net.load_state_dict(self.q_net.state_dict())
                        if self.verbose > 0:
                            if episode_rewards:
                                average_reward = sum(episode_rewards) / len(episode_rewards)
                                print(f"Total Steps: {total_steps}, Epsilon: {epsilon:.4f}, Average Episode Reward: {average_reward:.2f}")
                                if self.writer:
                                    self.writer.add_scalar("Average Episode Reward", average_reward, total_steps)
                                    self.writer.add_scalar("Epsilon", epsilon, total_steps)
                            else:
                                print(f"Total Steps: {total_steps}")
                except queue.Empty:
                    continue
        
        except KeyboardInterrupt:
            print("\nTraining interrupted. Cleaning up resources...")
        
        finally:
            print("Stopping workers and cleaning up resources...")
            stop_event.set()
            
            time.sleep(1)

            for conn in main_send_conns:
                conn.close()
            
            try:
                while True:
                    transition_queue.get(block=False)
            except queue.Empty:
                pass
            
            for w in workers:
                if w.is_alive():
                    w.terminate()
            
            for w in workers:
                w.join(timeout=0.5)
            
            if self.writer:
                self.writer.close()
                
            print("Training complete. Total steps: ", total_steps)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

if __name__ == "__main__":
    env_name = "LunarLander-v3"

    model = DQN(
        env_name=env_name,
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=4096,
        gamma=0.8,
        exploration_steps=1e5,
        exploration_final_eps=0.10,
        verbose=1,
        tensorboard_log="logs/",
        num_workers=32
    )
    print("Torch available:", torch.cuda.is_available())
    model.learn(int(5e12))
