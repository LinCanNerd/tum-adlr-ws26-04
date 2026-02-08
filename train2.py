"""
RMA Phase 2 Training: Adaptation Module Training

In this phase:
- The policy (actor) and privileged encoder are frozen (loaded from Phase 1)
- The adaptation module is trained to regress from stacked observations 
  to match the embedding produced by the privileged encoder

Usage:
    python train2.py --task=T1 --encoder=logs/2024-xx-xx/nn/model_1000.pth
    python train2.py --task=T1 --encoder=-1  # loads latest checkpoint
"""

import isaacgym
import os
import glob
import yaml
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
from utils.model import RMA
from utils.buffer import ExperienceBuffer
from utils.recorder import Recorder
from utils.wrapper import ObservationsWrapper
from envs import *


class AdaptationRunner:

    def __init__(self):
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()
        
        # Initialize environment (same as Runner)
        task_class = eval(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)
        self.env = ObservationsWrapper(self.env, self.cfg["runner"]["num_stack"])

        self.device = self.cfg["basic"]["rl_device"]
        
        # Initialize model (same architecture as Phase 1)
        self.model = RMA(
            self.env.num_actions, 
            self.env.num_obs, 
            obs_stacking=self.cfg["runner"]["num_stack"],
            num_privileged_obs=self.env.num_privileged_obs, 
            num_embedding=self.cfg["algorithm"]["num_embedding"]
        ).to(self.device)
        
        # Load pretrained encoder and policy from Phase 1
        self._load_encoder()
        
        # Freeze actor, critic, privileged encoder, and logstd
        self._freeze_ac_parameters()
        
        # Only optimize adaptation module
        self.learning_rate = self.cfg["algorithm"].get("adapt_learning_rate", 1e-3)
        self.optimizer = torch.optim.Adam(
            self.model.adapt_parameters(), 
            lr=self.learning_rate
        )

        # Setup buffer (similar to Runner)
        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("mirrored_obses", (self.env.num_obs,))
        self.buffer.add_buffer("mirrored_privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("stacked_obses",(self.env.num_stack, self.env.num_obs))
        self.buffer.add_buffer("mirrored_stacked_obses",(self.env.num_stack, self.env.num_obs))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)

    def _get_args(self):
        parser = argparse.ArgumentParser(description="RMA Phase 2: Adaptation Module Training")
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--encoder", required=True, type=str, 
                          help="Path to Phase 1 checkpoint. Use -1 for latest.")
        parser.add_argument("--num_envs", type=int, help="Number of environments. Overrides config.")
        parser.add_argument("--headless", type=bool, help="Run headless. Overrides config.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation.")
        parser.add_argument("--rl_device", type=str, help="Device for RL algorithm.")
        parser.add_argument("--seed", type=int, help="Random seed.")
        parser.add_argument("--max_iterations", type=int, help="Maximum training iterations.")
        self.args = parser.parse_args()

    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # Override with command line args (same logic as Runner)
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                elif arg != "encoder":  # encoder is handled separately
                    self.cfg["basic"][arg] = getattr(self.args, arg)
        
        # Disable video recording during training
        self.cfg["viewer"]["record_video"] = False

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))

        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _load_encoder(self):
        encoder_path = self.args.encoder
        
        # Handle -1 for latest checkpoint (same as Runner)
        if encoder_path == "-1" or encoder_path == -1:
            encoder_path = sorted(
                glob.glob(os.path.join("logs", "**/*.pth"), recursive=True), 
                key=os.path.getmtime
            )[-1]
        
        print(f"Loading Phase 1 model from {encoder_path}")
        checkpoint = torch.load(encoder_path, map_location=self.device, weights_only=True)
        
        # Load only actor, critic, privileged_encoder, and logstd
        # Exclude adaptation_module to train it from scratch
        model_state = checkpoint["model"]
        filtered_state = {k: v for k, v in model_state.items() 
                         if not k.startswith("adaptation_module")}
        
        self.model.load_state_dict(filtered_state, strict=False)
        print("Loaded actor, critic, and privileged encoder from Phase 1")
        
        # Load curriculum if available
        try:
            self.env.curriculum_prob = checkpoint.get("curriculum", self.env.curriculum_prob)
        except Exception as e:
            print(f"Note: Could not load curriculum: {e}")

    def _freeze_ac_parameters(self):
        """Freeze everything except adaptation module"""
        for param in self.model.critic.parameters():
            param.requires_grad = False
        for param in self.model.actor.parameters():
            param.requires_grad = False
        for param in self.model.privileged_encoder.parameters():
            param.requires_grad = False
        self.model.logstd.requires_grad = False
        
        # Print parameter counts per module
        critic_params = sum(p.numel() for p in self.model.critic.parameters())
        actor_params = sum(p.numel() for p in self.model.actor.parameters())
        encoder_params = sum(p.numel() for p in self.model.privileged_encoder.parameters())
        adapt_params = sum(p.numel() for p in self.model.adaptation_module.parameters())
        logstd_params = self.model.logstd.numel()
        
        print(f"Critic parameters:      {critic_params:,}")
        print(f"Actor parameters:       {actor_params:,}")
        print(f"Priv. encoder params:   {encoder_params:,}")
        print(f"Adaptation params:      {adapt_params:,}")
        print(f"Logstd parameters:      {logstd_params:,}")
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters:   {trainable:,} / {total:,}")

    def train(self):
        self.recorder = Recorder(self.cfg)
        
        # Reset environment
        obs, rew, done, infos = self.env.reset()
        obs = obs.to(self.device)
        done = done.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        stacked_obs = infos["stacked_obs"].to(self.device)
        
        for it in range(self.cfg["basic"]["max_iterations"]):
            # Collect rollout data
            for n in range(self.cfg["runner"]["horizon_length"]):
                mirrored_obs = self.env.mirror_obs(obs)
                mirrored_privileged_obs = self.env.mirror_priv(privileged_obs)
                mirrored_stacked_obs = self.env.mirror_obs(stacked_obs)

                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                self.buffer.update_data("stacked_obses", n, stacked_obs)
                self.buffer.update_data("mirrored_obses", n, mirrored_obs)
                self.buffer.update_data("mirrored_privileged_obses", n, mirrored_privileged_obs)
                self.buffer.update_data("mirrored_stacked_obses", n, mirrored_stacked_obs)
                                
                # Get action from frozen policy using privileged encoder
                with torch.no_grad():
                    dist, _ = self.model.act(obs, privileged_obs=privileged_obs)
                    
                    # Symmetric action (same as Phase 1)
                    mirrored_obs = self.env.mirror_obs(obs)
                    mirrored_privileged_obs = self.env.mirror_priv(privileged_obs)
                    mirrored_dist, _ = self.model.act(mirrored_obs, privileged_obs=mirrored_privileged_obs)
                    act = 0.5 * (dist.loc + self.env.mirror_act(mirrored_dist.loc)) + dist.scale * torch.randn_like(dist.loc)
                
                # Step environment
                obs, rew, done, infos = self.env.step(act)
                obs = obs.to(self.device)
                rew = rew.to(self.device)
                done = done.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                stacked_obs = infos["stacked_obs"].to(self.device)
                self.buffer.update_data("actions", n, act)
                self.buffer.update_data("rewards", n, rew)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("time_outs", n, infos["time_outs"].to(self.device))

                ep_info = {"reward": rew}
                ep_info.update(infos["rew_terms"])
                self.recorder.record_episode_statistics(
                    done, ep_info, it, 
                    n == (self.cfg["runner"]["horizon_length"] - 1)
                )

            # Train adaptation module
            mean_adapt_loss = 0.0
            mean_cosine_sim = 0.0
            mean_symmetric_loss = 0.0
            
            for _ in range(self.cfg["runner"]["mini_epochs"]):
                # Get target embedding from frozen privileged encoder
                with torch.no_grad():
                    _, target_embedding = self.model.act(self.buffer["obses"], privileged_obs=self.buffer["privileged_obses"])
                    _, mirrored_target_embedding = self.model.act(self.buffer["mirrored_obses"], privileged_obs=self.buffer["mirrored_privileged_obses"])
                
                dist, predicted_embedding = self.model.act(self.buffer["obses"], stacked_obs=self.buffer["stacked_obses"])
                mirrored_dist, mirrored_predicted_embedding = self.model.act(self.buffer["mirrored_obses"], stacked_obs=self.buffer["mirrored_stacked_obses"])
                mirrored_act = self.env.mirror_act(mirrored_dist.loc)

                # MSE loss
                adapt_loss = F.mse_loss(predicted_embedding, target_embedding)
                mirrored_adapt_loss = F.mse_loss(mirrored_predicted_embedding, mirrored_target_embedding)
                
                symmetric_loss = F.mse_loss(dist.loc, mirrored_act)
                tot_adapt_loss = (adapt_loss + mirrored_adapt_loss)/2

                tot_loss = symmetric_loss + tot_adapt_loss

                self.optimizer.zero_grad()
                tot_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.adapt_parameters(), 1.0)
                self.optimizer.step()
                
                mean_adapt_loss += tot_adapt_loss.item()
                mean_symmetric_loss += symmetric_loss.item()
                
                with torch.no_grad():
                    cosine_sim = F.cosine_similarity(
                        predicted_embedding.flatten(0, 1), 
                        target_embedding.flatten(0, 1), 
                        dim=-1
                    ).mean()
                    mean_cosine_sim += cosine_sim.item()
            
            mean_adapt_loss /= self.cfg["runner"]["mini_epochs"]
            mean_cosine_sim /= self.cfg["runner"]["mini_epochs"]
            mean_symmetric_loss /= self.cfg["runner"]["mini_epochs"]
                
            # Record statistics
            self.recorder.record_statistics(
                {
                    "adapt/loss": mean_adapt_loss,
                    "adapt/symmetric_loss": mean_symmetric_loss,
                    "adapt/cosine_sim": mean_cosine_sim,
                    "adapt/lr": self.learning_rate,
                    "curriculum/mean_lin_vel_level": self.env.mean_lin_vel_level,
                    "curriculum/mean_ang_vel_level": self.env.mean_ang_vel_level,
                    "curriculum/max_lin_vel_level": self.env.max_lin_vel_level,
                    "curriculum/max_ang_vel_level": self.env.max_ang_vel_level,
                },
                it,
            )

            # Save checkpoint
            if (it + 1) % self.cfg["runner"]["save_interval"] == 0:
                self.recorder.save(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "curriculum": self.env.curriculum_prob,
                    },
                    it + 1,
                )
            
            print("epoch: {}/{}".format(it + 1, self.cfg["basic"]["max_iterations"]))


if __name__ == "__main__":
    runner = AdaptationRunner()
    runner.train()