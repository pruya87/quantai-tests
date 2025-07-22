import retro
import numpy as np
import imageio
#import keyboard  
import time
import os
import glob
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import cv2
from collections import deque
from itertools import combinations, product

rendering = False

frames = []
max_reward = 0
last_r_toggle = 0
last_g_press = 0
rounds = 0
valid_combos = []
round_log = []
timestamp = time.localtime()
ts_format = f"{timestamp.tm_year}-{timestamp.tm_mon:02d}-{timestamp.tm_mday:02d}_{timestamp.tm_hour:02d}-{timestamp.tm_min:02d}"
observation_df = pd.DataFrame()

# Define your config constants (Separate in config.py?)
MAX_BUTTONS = 3

MOVEMENT_IDX = [4, 5, 6, 7]     # UP, DOWN, LEFT, RIGHT
ATTACK_IDX = [10, 1, 11, 8]     # X, A, Z, C
ATTACK_IDX = [10, 1, 11, 8]     # X, A, Z, C
DEFENSE_IDX = [9, 0, 3]         # Y, B, START

BUTTONS = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

BUTTONS_MAP = {
    'B'     : 'Block',
    'A'     : 'Low Punch',
    'MODE'  : 'Unused/Mode',
    'START' : 'Start/Block',
    'UP'    : 'Jump',
    'DOWN'  : 'Crouch',
    'LEFT'  : 'Move Left',
    'RIGHT' : 'Move Right',
    'C'     : 'Low Kick',
    'Y'     : 'Block',
    'X'     : 'High Punch',
    'Z'     : 'High Kick'
}



"""
SPECIAL_MOVES = {
    "Ice Ball":       {5, 7, 1},                # Down, Forward, Low Punch
    "Ice Puddle":     {5, 6, 8},                # Down, Back, Low Kick
    "Slide":          {1, 8, 0, 6},             # Low Punch, Low Kick, Block (B=0), Back
    "Uppercut":       {5, 10},                   # Down, High Punch
    "Sweep Leg":      {6, 8},                    # Back, Low Kick
    "Roundhouse":      {6, 11},                    # Back, High Kick
    "Jump Kick":      {4, 11},                   # Up, High Kick
    "Jump Forward Kick": {4, 7, 11},             # Up, Forward, High Kick
    "Jump Punch": {4, 10},                   # Up, High Punch
    "Jump Forward Punch": {4, 7, 10}         # Up, Forward, High Punch
}
"""

COMBO_DIFFICULTY = [
    'Roundhouse',
    'Sweep Leg',
    'Jump Punch',
    'Jump Forward Punch',
    'Jump Kick',
    'Ice Puddle',
    'Jump Forward Kick',
    'Uppercut',
    'Slide',
    'Ice Ball'
]

ROM_NAME="MortalKombatII-Genesis"
SAVE_STATE="Level1.SubZeroVsRyden.state"
SHARED_DIR="/workspace/shared/"
GIF_DIR="/workspace/shared/gifs/"
CKPT_DIR="/workspace/shared/checkpoints/"
RETRO_DIR="/usr/local/lib/python3.10/site-packages/retro/data/stable/"
MAX_EPISODES = 50  # Track 30 rounds
VISUAL_MODE = False
P1_ALGORITHM = 'dqn'
P1_PARAMETERS = {'learning_rate': 1e-3}

P2_ALGORITHM = 'random'
P2_PARAMETERS = {}


#######################################  AGENTS #######################################




class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()
        
class DQNAgent:
    def __init__(self, obs_space, action_space, learning_rate=1e-3):
        self.obs_space = obs_space
        self.action_space = action_space
        self.learning_rate = learning_rate

        # Epsilon-greedy settings
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = deque(maxlen=10000)

        # Very basic network, assumes flat input. Replace for convs.
        in_features = int(np.prod(obs_space.shape))
        self.q_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n)
        ).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, state):
        if np.random.rand() < self.epsilon:
            # --- EXPLORATION ---
            # Random action: limit number of buttons pressed
            """action = [0] * self.action_space.n
            pressed = random.sample(range(self.action_space.n), k=MAX_BUTTONS)
            
            for i in pressed:
                action[i] = 1"""
            action = random.choice(valid_combos)
            return action

        else:
            # --- EXPLOITATION ---
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)

            raw_action = (q_values > 0).int().cpu().tolist()[0]

            if sum(raw_action) > MAX_BUTTONS:
                top_idxs = torch.topk(q_values[0], MAX_BUTTONS).indices.tolist()
                action = [1 if i in top_idxs else 0 for i in range(len(raw_action))]
            else:
                action = raw_action

            return action
            
    def train(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)  # shape: (batch, 12)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Compute target Q-values for each bit (shape: batch_size x 12)
        with torch.no_grad():
            next_q_vals = self.q_network(next_states)  # (batch_size, 12)
            max_next_q_vals, _ = next_q_vals.max(dim=1, keepdim=True)  # max Q per batch (optional for stability)
            targets = rewards + self.gamma * max_next_q_vals * (1 - dones)  # shape (batch, 1)

            # Broadcast targets to all bits (optional, or just use per-bit targets)
            targets = targets.expand_as(next_q_vals)  # (batch_size, 12)

        # Current Q estimates for all bits
        curr_q_vals = self.q_network(states)  # (batch_size, 12)

        # Calculate loss over all bits (element-wise MSE)
        loss = self.criterion(curr_q_vals, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def save(self, filepath):
        torch.save(self.q_network.state_dict(), filepath)

    def load(self, filepath):
        self.q_network.load_state_dict(torch.load(filepath))
        self.q_network.eval()


class AgentManager:
    def __init__(self, env, model_name='random', model_params=None, visual_mode=VISUAL_MODE):
        self.env = env
        self.visual_mode = visual_mode
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = self._init_model(model_name, self.model_params)
        

    def _init_model(self, name, params):
        if name == 'random':
            return RandomAgent(self.env.action_space)
        elif name == 'dqn':
            return DQNAgent(self.env.observation_space, self.env.action_space, **params)
        else:
            raise ValueError(f"Unknown model name: {name}")

    def act(self, observation):
        return self.model.act(observation)

    def train(self, *args, **kwargs):
        if hasattr(self.model, 'train'):
            return self.model.train(*args, **kwargs)
            
    def save(self, path="."):
        if self.model_name != 'random':
            filename = os.path.join(path, f"{ts_format}_{self.model_name}.pth")  
            print(f"Saving model: {filename}")
            self.model.save(filename)
        else:
            print("Random agent has no model to save.")

    def load(self, path="."):
        if self.model_name != 'random':
            pattern = os.path.join(path, f"*_{self.model_name}.pth")
            files = glob.glob(pattern)
            if not files:
                print("No saved model found.")
                return
            latest_file = max(files, key=os.path.getmtime)
            print(f"Loading model from {latest_file}")
            self.model.load(latest_file)
        else:
            print("Random agent has no model to load.")

################################ CUSTOM WRAPPERS AND HELPERS ###############################

def render_with_info(frame, info_dict, position=(10, 30), font_scale=0.3, color=(0,255,0), thickness=1):
    # frame: np.ndarray (H, W, 3) BGR image
    # info_dict: dictionary with keys and values to display
    
    y_offset = position[1]
    for key, val in info_dict.items():
        text = f"{key}: {val}"
        cv2.putText(frame, text, (position[0], y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, color, thickness, cv2.LINE_AA)
        y_offset += int(25 * font_scale)  # move down for next line

    return frame

def save_video(frames, filename, gray=False, fps=30):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' or 'XVID'
    video = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        # If frames are grayscale, convert to BGR first:
        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Transform BGR (OpenCV native) in RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.write(frame_rgb)

    video.release()
    
# Resize to half resolution or a fixed small size
def compress_img(frame, scale=0.5):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w*scale), int(h*scale)))
    
def export_observation():
    filename = os.path.join(CKPT_DIR, f"{ts_format}_{P1_ALGORITHM}_{ROM_NAME}.csv")  
    print(f"Exporting training logs: {filename}")
    observation_df.to_csv(filename, index=False)

def get_special_moves(orientation='right'):
    if orientation == 'right':
        forward = 7
        backward = 6
    else:
        forward = 6
        backward = 7

    SPECIAL_MOVES = {
        "Ice Ball":       {5, forward, 1},              # Down, Forward, Low Punch
        "Ice Puddle":     {5, backward, 8},             # Down, Back, Low Kick
        "Slide":          {1, 8, 0, backward},          # Low Punch, Low Kick, Block (B=0), Back
        "Uppercut":       {5, 10},                       # Down, High Punch
        "Sweep Leg":      {backward, 8},                 # Back, Low Kick
        "Roundhouse":     {backward, 11},                # Back, High Kick
        "Jump Kick":      {4, 11},                        # Up, High Kick
        "Jump Forward Kick": {4, forward, 11},           # Up, Forward, High Kick
        "Jump Punch":     {4, 10},                        # Up, High Punch
        "Jump Forward Punch": {4, forward, 10}           # Up, Forward, High Punch
    }

    return SPECIAL_MOVES
    
def detect_special_move(action, SPECIAL_MOVES, base_reward=1.5):
    pressed_buttons = {i for i, pressed in enumerate(action) if pressed == 1}
    for move_name, combo_buttons in SPECIAL_MOVES.items():
        if combo_buttons.issubset(pressed_buttons):
            if move_name in COMBO_DIFFICULTY:
                difficulty = COMBO_DIFFICULTY.index(move_name)
                return move_name, base_reward * (difficulty + 1)
            else:
                return move_name, base_reward  # fallback
    return None, 0
    
def combo_generator():
    # Indices from your mapping
    movement_buttons = MOVEMENT_IDX  # [4, 5, 6, 7]
    attack_defense_buttons = ATTACK_IDX + DEFENSE_IDX  # e.g. [10,1,11,8,9,0,3]

    # All combos with up to 2 movement buttons:
    movement_combos = []
    for r in range(0, 3):  # 0, 1 or 2 movement buttons
        movement_combos.extend(combinations(movement_buttons, r))

    # All combos with up to 1 attack/defense button:
    attack_defense_combos = []
    for r in range(0, 2):  # 0 or 1 attack/defense buttons
        attack_defense_combos.extend(combinations(attack_defense_buttons, r))
        
    # Combine them into full action combos
    valid_combos = []
    for mov in movement_combos:
        for ad in attack_defense_combos:
            combo = [0] * 12  # total buttons count
            for m in mov:
                combo[m] = 1
            for a in ad:
                combo[a] = 1
            valid_combos.append(combo)
    return valid_combos

def decode_action_vector(action_vector):
    return [BUTTONS_MAP[btn] for btn, pressed in zip(BUTTONS, action_vector) if pressed]
    
def extract_valid_actions(env, sample_count=10000):
    test = []
    for _ in range(sample_count):
        test.append(env.action_space.sample().tolist())
    unique_actions = list(map(list, set(map(tuple, test))))
    return unique_actions

def convert_int_to_action(n, action_size=12):
    """Convert integer n to a 12-bit binary list."""
    return [int(b) for b in f"{n:0{action_size}b}"]

def check_render_toggle():
    try:
        with open("/workspace/shared/render.txt", "r") as f:
            return f.read().strip() == "1"
    except FileNotFoundError:
        return False
        
        
class InputSwitcherWrapper(gym.ObservationWrapper):
    def __init__(self, env, visual_mode=VISUAL_MODE):
        super().__init__(env)
        self.visual_mode = visual_mode

    def observation(self, obs):
        if self.visual_mode:
            frame = self.env.render(mode="rgb_array")
            #print("[InputSwitcherWrapper] Returning preprocessed frame.")
            return self.preprocess_frame(frame)
        #print("[InputSwitcherWrapper] Returning original obs.")
        return obs

    def preprocess_frame(self, frame):
        # Example: grayscale, resize, normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))  # Standard size for Atari-like models
        frame = frame.astype(np.float32) / 255.0
        return np.expand_dims(frame, axis=0)  # shape: (1, 84, 84)
        
class EnvMemoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.player_rounds_won = 0
        self.enemy_rounds_won = 0

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # Support new gym reset returning (obs, info)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        info['player_rounds_won'] = self.player_rounds_won
        info['enemy_rounds_won'] = self.enemy_rounds_won
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if terminated or truncated:
            if info.get('health', 0) > info.get('enemy_health', 0):
                self.player_rounds_won += 1
            else:
                self.enemy_rounds_won += 1

        info['player_rounds_won'] = self.player_rounds_won
        info['enemy_rounds_won'] = self.enemy_rounds_won
        info['rounds_won'] = self.enemy_rounds_won + self.player_rounds_won
        
        return obs, reward, terminated, truncated, info

    def reset_counters(self):
        self.player_rounds_won = 0
        self.enemy_rounds_won = 0
 
class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_health = None
        self.previous_enemy_health = None
        self.previous_enemy_y = None
        self.total_reward = 0   

    def reset(self, **kwargs):
        self.total_reward = 0
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        print("=" * 20)
        print(f"Obs after reset (type): {type(obs)}")  # should be tuple here
        print("=" * 20)
        
        self.previous_health = info.get('health', 0)
        self.previous_enemy_health = info.get('enemy_health', 0)
        self.previous_enemy_y = info.get('enemy_y_position', 0)
        
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        print(f"Obs reward wrapper in: {type(obs)}")
        self.log_round = info.copy()

        # === Health tracking ===
        current_health = info.get('health', 0)
        current_enemy_health = info.get('enemy_health', 0)

        delta_health = self.previous_health - current_health
        delta_enemy_health = self.previous_enemy_health - current_enemy_health
        
        #self.log_round['delta_health'] = delta_health
        #self.log_round['delta_enemy_health'] = delta_enemy_health
        
        self.attack_received = 15 < delta_health < 25 and not 120
        self.attack_dealt = 15 < delta_enemy_health < 25 and not 120
        self.attack_block = delta_health < 15 or delta_enemy_health < 15
        
        custom_reward = int(self.attack_dealt) * 25 - int(self.attack_received) * 25 + int(self.attack_block) * 15
        
        
        custom_reward += int((2 * delta_enemy_health - delta_health))
        self.log_round['damage_reward'] = custom_reward

        # === Threshold bonuses ===
        if self.previous_enemy_health > 80 > current_enemy_health:
            custom_reward += 50
            self.log_round['threshold_reward'] = 50
        elif self.previous_enemy_health > 50 > current_enemy_health:
            custom_reward += 100
            self.log_round['threshold_reward'] = 150
        elif self.previous_enemy_health > 25 > current_enemy_health:
            custom_reward += 200
            self.log_round['threshold_reward'] = 350
        elif self.previous_enemy_health > 5 > current_enemy_health:
            custom_reward += 300
            self.log_round['threshold_reward'] = 350
        else:
            self.log_round['threshold_reward'] = 0

        self.previous_health = current_health
        self.previous_enemy_health = current_enemy_health

        # === Survival reward ===
        if not terminated:
            custom_reward += 0.25
            self.log_round['survival_reward'] = 0.25
        elif current_health > 0:
            custom_reward += 800
            self.log_round['survival_reward'] = 800
        else:
            custom_reward -= 600
            self.log_round['survival_reward'] = -600

        self.total_reward += custom_reward
        self.log_round['reward'] = custom_reward
        self.log_round['total_reward'] = self.total_reward

        # === NEW: Orientation & Distance Awareness ===
        agent_x = info.get('x_position', 0)
        agent_y = info.get('y_position', 0)
        enemy_x = info.get('enemy_x_position', 0)
        enemy_y = info.get('enemy_y_position', 0)

        orientation = 'right' if agent_x < enemy_x else 'left'
        distance = abs(agent_x - enemy_x)
        enemy_y_delta = enemy_y - self.previous_enemy_y
        self.previous_enemy_y = enemy_y

        self.log_round.update({
            'orientation': orientation,
            'distance_to_enemy': distance,
            'enemy_y_movement': enemy_y_delta,
            'enemy_jumping': enemy_y_delta > 5,
            'enemy_falling': enemy_y_delta < -5,
        })
        
        # === Input debugging ===
        
        input_vector = action  # this is the 12-bit MultiBinary vector
        button_names = [env.buttons[i] for i, v in enumerate(input_vector) if v]   
        
        alias_map = {
            'UP': '^', 'DOWN': 'v', 'LEFT': '<', 'RIGHT': '>',
            'A': 'LP', 'B': 'BL', 'C': 'LK', 'X': 'HP', 'Y': 'BL', 'Z': 'HK'
        }       
        
        if info.get('Recovery', 0):
            self.log_round['Input'] = [0] * 12
        else:
            self.log_round['Input'] = [alias_map.get(b, b) for b in button_names]
        
        # === Combo detection ===
        SPECIAL_MOVES = get_special_moves(orientation=orientation)
        special_move, combo_reward = detect_special_move(action, SPECIAL_MOVES)
        
  
            
        if special_move and agent_y == 0:
            custom_reward += combo_reward
            self.log_round['special_move'] = special_move
            self.log_round['combo_reward'] = combo_reward
        else:
            self.log_round['special_move'] = 'None'
            self.log_round['combo_reward'] = 0
        
        #print(f"Obs reward wrapper out: {type(obs)}")
        return obs, custom_reward, terminated, truncated, info

    @property
    def log_step(self):
        return self.log_round
        
class RecoveryManager:
    def __init__(self, standup_duration=4):
        self.in_arc = False
        self.in_standup = False
        self.standup_timer = 0

    def update(self, was_hit, y):
        if was_hit and abs(y) > 0 and not self.in_arc:
            #print("→ Entering arc recovery")
            self.in_arc = True
            self.in_standup = False

        if self.in_arc and abs(y) == 0:
            #print("→ Landed, starting standup")
            self.in_arc = False
            self.in_standup = True
            self.standup_timer = 4  # adjust as needed

        if self.in_standup:
            #print(f"→ Standup recovery: {self.standup_timer} frames left")
            self.standup_timer -= 1
            if self.standup_timer <= 0:
                #print("→ Fully recovered")
                self.in_standup = False

    def in_recovery(self):
        return self.in_arc or self.in_standup
        
class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self.recovery = RecoveryManager()
        self.last_y = 0
        self.previous_health = 0
        self.last_hit = False
        self.n_actions = self.action_space.shape[0]

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        self.previous_health = info.get('health', 0)
        
        return obs, info

    def get_agent_y(self, obs, info):
        return info.get('y_position', 0)

    def detect_hit(self, obs, info):
        return info.get('health', 0) < self.previous_health

    def step(self, action):
        done = False
        truncated = False
        info = {}
        obs = None
        reward = 0

        for _ in range(self._skip):
            # Update recovery status before sending action
            self.recovery.update(self.last_hit, self.last_y)

            if self.recovery.in_recovery():
                action_to_send = [0] * self.n_actions
            else:
                action_to_send = action
                

            obs, reward, done, truncated, info = self.env.step(action_to_send)

            self.last_y = self.get_agent_y(obs, info)
            self.last_hit = self.detect_hit(obs, info)

            if done or truncated:
                break
        
        self.previous_health = info.get('health', 0)
        info['Recovery'] = self.recovery.in_recovery()
        
        return obs, reward, done, truncated, info

class FrameStackPixelsOnly(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        # We won't change observation_space here because env expects pixels only
        # So stacked pixels will have shape (224, 320, 3*k) but observation_space still says (224,320,3)
        # Be careful: some code that uses observation_space might break or misinterpret shape

    def reset(self):
        obs = self.env.reset()
        pixels, info_dict = obs
        for _ in range(self.k):
            self.frames.append(pixels)
        stacked_pixels = self._get_stacked_frames()
        return (stacked_pixels, info_dict)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pixels, info_dict = obs
        self.frames.append(pixels)
        stacked_pixels = self._get_stacked_frames()
        done = terminated or truncated
        return (stacked_pixels, info_dict), reward, terminated, truncated, info

    def _get_stacked_frames(self):
        return np.concatenate(list(self.frames), axis=2)

        
class Init_sandbox():
    def __init__(self):
        self.env = retro.make(game=ROM_NAME, state=SAVE_STATE)
        #self.env = InputSwitcherWrapper(self.env, visual_mode=VISUAL_MODE) 
        #self.env = EnvMemoryWrapper(self.env)
        #self.env = FrameSkipWrapper(self.env, skip=4)
        self.env = CustomRewardWrapper(self.env)
        self.valid_combos = combo_generator()
        
        # Instantiate agents from config
        self.P1_agent = AgentManager(self.env, model_name=P1_ALGORITHM, model_params=P1_PARAMETERS)
        #P2_agent = AgentManager(env, model_name=P2_ALGORITHM, model_params=P2_PARAMETERS)
       
sandbox = Init_sandbox()
env = sandbox.env
P1_agent = sandbox.P1_agent
obs = env.reset()


try:
    print("Observation space:", env.observation_space)
    print("Observation pixels shape:", obs[0].shape)
    print("Observation pixels dtype:", obs[0].dtype)
    print("Observation info dict:", obs[1])
except:
    print("Error loading ")

    
valid_combos = sandbox.valid_combos

try:
    P1_agent.load(f"{CKPT_DIR}")
    print("Model loaded")
except:
    print("Model failed to load")

#step_result = env.step(env.action_space.sample())
#print(f"Type: {type(step_result)}")
#print(f"Length: {len(step_result)}")
#print(f"Content: {step_result}")

frames = []

while rounds < MAX_EPISODES:    
   
    # Total time elapsed since the timer started
    #roundtime = round((time.time() - starttime), 2)
    
    """now = time.time()

    if keyboard.is_pressed('r') and now - last_r_toggle > 0.5:
        rendering = not rendering
        last_r_toggle = now
        print(f"Rendering toggled: {rendering}")
    
    if rendering:
        frame = env.get_screen()  # returns an RGB ndarray
        frames.append(frame)

    if keyboard.is_pressed('g') and now - last_g_press > 0.5 and frames:
        imageio.mimsave(f"{SHARED_DIR}/gif/{SAVE_STATE}-gameplay.gif", frames, fps=30)
        print(f"Saved {len(frames)} frames")
        frames.clear()
        last_g_press = now
        print("GIF saved.")"""
    
    frames.append(compress_img(env.get_screen(), scale=1))
   
    prev_obs = obs
    
    print(f"Observation in training loop before step: {type(obs)}")

    action = P1_agent.act(prev_obs)
    #print(decode_action_vector(action))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    print(f"Observation in training loop after step: {type(obs)}")
    
    log_step = env.log_step
    round_log.append(log_step)
    
    # Useful for random agent. DQN uses internal buffer
    #P1_agent.train(prev_obs, action, reward, obs, done)
    
    #P1_agent.train()
    
    if done:
        #starttime = time.time()
        print(f"Episode done. Total reward: {log_step['total_reward']}")
        P1_agent.train()
        if log_step['total_reward'] > max_reward:
            max_reward = log_step['total_reward']
               
        obs = env.reset()
        rounds += 1 
        print(f"New Round {rounds}: Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {log_step}")
        
print(f"Max reward: {max_reward}")
P1_agent.save(f"{CKPT_DIR}")   
# Save and export logs
observation_df = pd.DataFrame(round_log)
export_observation()     

for frame, log in zip(frames, round_log):
    frame = render_with_info(frame, log)
    
print(f"Saving renders as {GIF_DIR}{SAVE_STATE}-gameplay.mp4")
save_video(frames, f"{GIF_DIR}{SAVE_STATE}-gameplay.mp4", fps=24)
#imageio.mimsave(f"{GIF_DIR}{SAVE_STATE}-gameplay.gif", frames, fps=30)
env.close()