import numpy as np
from robot_arm_env import RobotArmEnv

def random_agent(env, num_episodes=3):
    # Enable human rendering mode
    env = RobotArmEnv(render_mode='human')

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0.0

        print(f"Episode {episode + 1}/{num_episodes} starting...")
        env.render()  # Render the initial state

        while not done:
            step_count += 1
            # Randomly select an action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Render the current state
            #env.render()  # Render the current state
            #print(f"--- Step {step_count:02d} | Action {action} | Step Reward: {reward:6.1f}")

        reason = "Collision/Success" if terminated else "Max Steps Reached"
        print(f"Episode {episode + 1} Finished in {step_count}steps | Total Reward: {total_reward} | Reason: {reason}\n")

if __name__ == "__main__":
    # Create the environment
    env = RobotArmEnv(render_mode='human')

    # Run the random agent
    random_agent(env, num_episodes=10)

    # Close the environment after running
    env.close()