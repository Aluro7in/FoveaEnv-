import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from rl_env import ForveaGymEnv

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    env = make_vec_env(lambda: ForveaGymEnv(task_id="medium"), n_envs=4)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Add entropy bonus for exploration
        tensorboard_log="./logs",
    )
    model.learn(total_timesteps=500_000)
    model.save("models/forvea_ppo")
    print("Training complete. Saved model to models/forvea_ppo.zip")
