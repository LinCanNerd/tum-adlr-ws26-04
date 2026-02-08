import isaacgym
import os
import glob
import yaml
import argparse
import numpy as np
import pandas as pd
import random
import torch
from utils.model import RMA
from utils.wrapper import ObservationsWrapper
from envs import *


class EmbeddingExtractor:

    def __init__(self):
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()

        # Initialize environment
        task_class = eval(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)
        self.env = ObservationsWrapper(self.env, self.cfg["runner"]["num_stack"])

        self.device = self.cfg["basic"]["rl_device"]

        # Initialize trained model
        self.model = RMA(
            self.env.num_actions,
            self.env.num_obs,
            obs_stacking=self.cfg["runner"]["num_stack"],
            num_privileged_obs=self.env.num_privileged_obs,
            num_embedding=self.cfg["algorithm"]["num_embedding"]
        ).to(self.device)

        # Initialize random model (for random embeddings)
        self.random_model = RMA(
            self.env.num_actions,
            self.env.num_obs,
            obs_stacking=self.cfg["runner"]["num_stack"],
            num_privileged_obs=self.env.num_privileged_obs,
            num_embedding=self.cfg["algorithm"]["num_embedding"]
        ).to(self.device)

        # Load pretrained model
        self._load_model()

        # Freeze all models for inference only
        self.model.eval()
        self.random_model.eval()

        print(f"Embedding dimension: {self.cfg['algorithm']['num_embedding']}")

    def _get_args(self):
        parser = argparse.ArgumentParser(description="RMA Embedding Extraction")
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--model", required=True, type=str,
                          help="Path to trained checkpoint. Use -1 for latest.")
        parser.add_argument("--num_samples", type=int, default=20000,
                          help="Number of samples to collect.")
        parser.add_argument("--sample_interval", type=int, default=57,
                          help="Save embeddings every N steps (default: 10).")
        parser.add_argument("--output", type=str, default="embeddings.csv",
                          help="Output CSV file name.")
        parser.add_argument("--num_envs", type=int, help="Number of environments. Overrides config.")
        parser.add_argument("--headless", type=bool, help="Run headless. Overrides config.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation.")
        parser.add_argument("--rl_device", type=str, help="Device for RL algorithm.")
        parser.add_argument("--seed", type=int, help="Random seed.")
        self.args = parser.parse_args()

    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        # Override with command line args
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                elif arg not in ["model", "num_samples", "output"]:  # handled separately
                    self.cfg["basic"][arg] = getattr(self.args, arg)

        # Force headless for efficiency
        self.cfg["basic"]["headless"] = True
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

    def _load_model(self):
        model_path = self.args.model

        # Handle -1 for latest checkpoint
        if model_path == "-1" or model_path == -1:
            model_path = sorted(
                glob.glob(os.path.join("logs", "**/*.pth"), recursive=True),
                key=os.path.getmtime
            )[-1]

        print(f"Loading trained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Load the full trained model
        self.model.load_state_dict(checkpoint["model"])
        print("Loaded trained model successfully")

        # For random model, only load actor and privileged_encoder (not adaptation_module)
        # This ensures the adaptation module produces random embeddings
        model_state = checkpoint["model"]
        filtered_state = {k: v for k, v in model_state.items()
                         if not k.startswith("adaptation_module")}
        self.random_model.load_state_dict(filtered_state, strict=False)
        print("Initialized random model (untrained adaptation module)")

    def extract_embeddings(self):
        """Extract embeddings and save to CSV"""
        print(f"\nCollecting {self.args.num_samples} samples (every {self.args.sample_interval} steps)...")

        # Storage for embeddings
        target_embeddings = []
        predicted_embeddings = []
        random_embeddings = []

        # Reset environment
        obs, rew, done, infos = self.env.reset()
        obs = obs.to(self.device)
        privileged_obs = infos["privileged_obs"].to(self.device)
        stacked_obs = infos["stacked_obs"].to(self.device)

        samples_collected = 0
        step_counter = 0

        with torch.no_grad():
            while samples_collected < self.args.num_samples:
                # Only save embeddings every sample_interval steps
                if step_counter % self.args.sample_interval == 0:
                    # Get target embedding from privileged encoder
                    _, target_emb = self.model.act(obs, privileged_obs=privileged_obs)

                    # Get predicted embedding from trained adaptation module
                    _, predicted_emb = self.model.act(obs, stacked_obs=stacked_obs)

                    # Get random embedding from untrained adaptation module
                    _, random_emb = self.random_model.act(obs, stacked_obs=stacked_obs)

                    # Store embeddings (flatten batch dimension)
                    target_embeddings.append(target_emb.cpu().numpy())
                    predicted_embeddings.append(predicted_emb.cpu().numpy())
                    random_embeddings.append(random_emb.cpu().numpy())

                    samples_collected += obs.shape[0]

                    if samples_collected % 100 == 0:
                        print(f"Collected {samples_collected}/{self.args.num_samples} samples...")

                # Take action and step environment
                dist, _ = self.model.act(obs, privileged_obs=privileged_obs)
                act = dist.sample()

                obs, rew, done, infos = self.env.step(act)
                obs = obs.to(self.device)
                privileged_obs = infos["privileged_obs"].to(self.device)
                stacked_obs = infos["stacked_obs"].to(self.device)

                step_counter += 1

        # Concatenate all embeddings
        target_embeddings = np.concatenate(target_embeddings, axis=0)
        predicted_embeddings = np.concatenate(predicted_embeddings, axis=0)
        random_embeddings = np.concatenate(random_embeddings, axis=0)

        # Trim to exact number of samples
        target_embeddings = target_embeddings[:self.args.num_samples]
        predicted_embeddings = predicted_embeddings[:self.args.num_samples]
        random_embeddings = random_embeddings[:self.args.num_samples]

        print(f"\nCollected embeddings shape:")
        print(f"  Target: {target_embeddings.shape}")
        print(f"  Predicted: {predicted_embeddings.shape}")
        print(f"  Random: {random_embeddings.shape}")

        # Create DataFrame with clear labels
        embedding_dim = target_embeddings.shape[1]

        data = {}

        # Add target embeddings
        for i in range(embedding_dim):
            data[f'target_dim_{i}'] = target_embeddings[:, i]

        # Add predicted embeddings
        for i in range(embedding_dim):
            data[f'predicted_dim_{i}'] = predicted_embeddings[:, i]

        # Add random embeddings
        for i in range(embedding_dim):
            data[f'random_dim_{i}'] = random_embeddings[:, i]

        # Add sample index
        data['sample_index'] = np.arange(self.args.num_samples)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Reorder columns to have sample_index first
        cols = ['sample_index'] + [col for col in df.columns if col != 'sample_index']
        df = df[cols]

        # Save to CSV
        output_path = self.args.output
        df.to_csv(output_path, index=False)
        print(f"\nSaved embeddings to {output_path}")
        print(f"Total rows: {len(df)}")
        print(f"Total columns: {len(df.columns)} (1 index + {embedding_dim}*3 embedding dimensions)")

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"  Target embeddings - Mean: {target_embeddings.mean():.4f}, Std: {target_embeddings.std():.4f}")
        print(f"  Predicted embeddings - Mean: {predicted_embeddings.mean():.4f}, Std: {predicted_embeddings.std():.4f}")
        print(f"  Random embeddings - Mean: {random_embeddings.mean():.4f}, Std: {random_embeddings.std():.4f}")


if __name__ == "__main__":
    extractor = EmbeddingExtractor()
    extractor.extract_embeddings()
