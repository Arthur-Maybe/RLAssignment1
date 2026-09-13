import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RobotArmEnv(gym.Env):
    """
    Custom Robotic Arm Environment for Pick-and-Place Task
    """
    metadata = {'render_modes': ['human', "ansi"]}

    def __init__(self, render_mode=None):
        super().__init__()

        # Link lengths
        self.d1 = 1.0
        self.d2 = 1.5

        # Bolt pickup (sb, yb) and hole target (xh, yh)
        self.bolt_pos = np.array([2.176, 0.129])
        self.hole_pos = np.array([2.266, 0.643])

        # Joint Limits (in radians)
        self.min_theta1, self.max_theta1 = 0.0, np.pi / 3.0
        self.min_theta2, self.max_theta2 = -np.pi / 3.0, np.pi / 3.0
        self.detal_theta = np.pi / 9.0  # Increment for joint angles

        # Action Space
        # d_theta1 (0: -20 degrees, 1: 0 degrees, 2: +20 degrees)
        # d_theta2 (0: -20 degrees, 1: 0 degrees, 2: +20 degrees)
        # gripper (0: open, 1: close)
        self.action_space = spaces.MultiDiscrete([3, 3, 2])

        # Obversation Space
        # theta1, theta2, gripper_state, has_bolt
        low = np.array([self.min_theta1, self.min_theta2, 0, 0], dtype=np.float32)
        high = np.array([self.max_theta1, self.max_theta2, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Initialize state
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.gripper_state = 0  # 0: open, 1: closed
        self.has_bolt = 0  # 0: no bolt, 1: has bolt

        self.render_mode = render_mode

    def _get_obs(self):
        return np.array([self.theta1, self.theta2, float(self.gripper_state), float(self.has_bolt)], dtype=np.float32)

    def _get_gripper_pos(self):
        """
        Calculate the position of the gripper based on the current joint angles.
        Returns:
            np.array: The (x, y) position of the gripper.
        """
        xg = self.d1 * np.cos(self.theta1) + self.d2 * np.cos(self.theta1 + self.theta2)
        yg = self.d1 * np.sin(self.theta1) + self.d2 * np.sin(self.theta1 + self.theta2)
        return np.array([xg, yg])

def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    # Initial starting angles
    self.theta1 = 0.0
    self.theta2 = 0.0
    self.gripper_state = 0  # Open
    self.has_bolt = 0  # No bolt

    observation = self._get_obs()
    info = {}
    return observation, info

def step(self, action):
    # Unpack action choices
    # action[0]: theta1 delta
    # action[1]: theta2 delta
    # action[2]: gripper state
    d_theta1 = (action[0] - 1) * self.detal_theta  # -20, 0, +20 degrees
    d_theta2 = (action[1] - 1) * self.detal_theta
    new_gripper = action[2]

    # Update joint angles with limits
    self.theta1 = np.clip(self.theta1 + d_theta1, self.min_theta1, self.max_theta1)
    self.theta2 = np.clip(self.theta2 + d_theta2, self.min_theta2, self.max_theta2)
    self.gripper_state = new_gripper

    # Calculate gripper position
    gripper_pos = self._get_gripper_pos()

    # Define pickup and target tolerances
    pickup_tolerance = 0.1

    # Handle bolt pickup
    if self.has_bolt == 0 and self.gripper_state == 1:  # Gripper closed
        if np.linalg.norm(gripper_pos - self.bolt_pos) < pickup_tolerance:
            self.has_bolt = 1  # Successfully picked up the bolt

    # If gripper opens while holding the bolt, drop it
    if self.has_bolt == 1 and self.gripper_state == 0:  # Gripper opened
        self.has_bolt = 0  # Drop the bolt

    # Evaluate terminal conditions and rewards
    reward = -1.0
    terminated = False

    # Successful placement condition
    if self.has_bolt == 1 and np.linalg.norm(gripper_pos - self.hole_pos) < pickup_tolerance and self.gripper_state == 0:
        reward = 10.0
        terminated = True
    # Collision condition
    elif gripper_pos[0] == self.hole_pos[0] and gripper_pos[1] != self.hole_pos[1]:
        reward = -100.0
        terminated = True

    observation = self._get_obs()
    info = {"gripper_pos": gripper_pos}

    return observation, reward, terminated, False, info