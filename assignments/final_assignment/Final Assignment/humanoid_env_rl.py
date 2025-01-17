import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from simulation import Simulation

class HumanoidEnv(gym.Env):
    """
    Custom OpenAI Gym environment for controlling a humanoid's walk.
    """

    def __init__(self):
        super(HumanoidEnv, self).__init__()

        # Initialize the simulation
        self.simulation = Simulation()
        self.simulation_clock = pygame.time.Clock()

        # Define the action space: motor speeds for 4 joints (normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1]),  # Normalized action space
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )

        # Define the observation space based on the humanoid's log_state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.simulation.humanoid.log_state()),),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state and return the initial observation.
        """
        del self.simulation  # Clear the previous simulation
        self.simulation = Simulation()  # Restart the simulation
        return self._get_observation(), {}

    def step(self, action):
        """
        Perform one step in the environment with the given action.

        Args:
            action (np.array): Normalized motor speeds for the joints.

        Returns:
            observation (np.array): The new observation.
            reward (float): The computed reward.
            done (bool): Whether the episode is finished.
            info (dict): Additional debug information.
        """
        # Scale the action back to the motor speed range
        max_motor_speed = 10  # Adjust based on the motor's actual speed limits
        scaled_action = action * max_motor_speed
        self.simulation.humanoid.update_motors(scaled_action)

        # Step the simulation forward
        self.simulation.world.Step(1.0 / 60.0, 6, 2)

        # Get the new observation
        observation = self._get_observation()

        # Compute the reward
        reward = self._compute_reward()

        # Check if the episode is done
        done = self._is_done()

        # Additional info (can include debug data if needed)
        info = {}

        return observation, reward, done, False, info

    def render(self, mode='human'):
        """
        Render the environment using the simulation's rendering system.
        """
        self.simulation.screen.fill(self.simulation.bg_color)
        self.simulation.render_ground()
        self.simulation.render_flag()
        self.simulation.humanoid.render(self.simulation.screen, self.simulation.ppm)
        pygame.display.flip()
        self.simulation_clock.tick(600)

    def close(self):
        """
        Close the environment.
        """
        pygame.quit()

    def _get_observation(self):
        """
        Get the current observation from the humanoid's state.

        Returns:
            np.array: The state as a flattened array.
        """
        state = self.simulation.humanoid.log_state()
        return np.array(list(state.values()), dtype=np.float32)

    def _compute_reward(self):
        """
        Compute the reward based on the current state.

        Returns:
            float: The reward value.
        """
        state = self.simulation.humanoid.log_state()
        x = state.get('torso_x', 0)
        y = state.get('torso_y', 0)

        # Reward parameters
        straight_coef = 5000
        forward_coef = 10
        velocity_coef = 100
        done_bonus = 10000

        # Positivity function for velocity
        def pos(val):
            return val if val > 0 else -1

        # Compute reward
        reward = ((y - 1.7) * straight_coef +
                  (x - 2) * forward_coef +
                  (pos(state.get('left_thigh_vx')) +
                   pos(state.get('right_thigh_vx')) +
                   pos(state.get('right_shin_vx')) +
                   pos(state.get('left_shin_vx')) +
                   state.get('torso_vx')) * velocity_coef)

        if x > 14.5:  # Bonus for completing the task
            reward += done_bonus

        return reward

    def _is_done(self):
        """
        Determine if the episode is done.

        Returns:
            bool: True if the episode is finished, otherwise False.
        """
        state = self.simulation.humanoid.log_state()
        x = state.get('torso_x', 0)

        # End the episode if the humanoid reaches the goal
        return x > 14.5
