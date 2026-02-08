import torch
import torch.nn.functional as F


class RMA(torch.nn.Module):

    def __init__(self, num_act, num_obs, obs_stacking, num_privileged_obs, num_embedding):
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_obs + num_privileged_obs, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs + num_embedding, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_act),
        )
        self.privileged_encoder = torch.nn.Sequential(
            torch.nn.Linear(num_privileged_obs, num_embedding),
            torch.nn.ELU(),
            torch.nn.Linear(num_embedding, num_embedding),
        )
        # Convolutional adaptation module to preserve temporal structure
        # Using smaller filters (64, 32) to reduce memory with large batch sizes
        # Input shape: (batch, num_stack, num_obs) -> treat as (batch, num_obs, num_stack)
        self.adaptation_module = torch.nn.Sequential(
            # Transpose will happen in forward pass: (batch, num_stack, num_obs) -> (batch, num_obs, num_stack)
            torch.nn.Conv1d(num_obs, 32, kernel_size=3, padding=1),
            torch.nn.ELU(),
            torch.nn.Conv1d(32, 16, kernel_size=3, padding=1),
            torch.nn.ELU(),
            # Global average pooling over temporal dimension
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, num_embedding),
        )
        self.num_obs = num_obs
        self.obs_stacking = obs_stacking
        self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_act), fill_value=-2.0), requires_grad=True)

    def act(self, obs, privileged_obs = None, stacked_obs = None):
        if privileged_obs is not None:
            embedding = self.privileged_encoder(privileged_obs)
        elif stacked_obs is not None:
            # Handle 4D input from buffer: [H, N, stack, obs] -> [H*N, stack, obs]
            original_shape = None
            if stacked_obs.dim() == 4:
                original_shape = stacked_obs.shape[:2]  # Save (H, N) for reshaping back
                batch_shape = stacked_obs.shape[0] * stacked_obs.shape[1]
                stacked_obs = stacked_obs.view(batch_shape, stacked_obs.shape[2], stacked_obs.shape[3])

            # After reshape: [batch, dim1, dim2]
            # We need [batch, num_obs, num_stack] for Conv1d (channels, sequence_length)
            # If shape is [batch, num_obs, num_stack], keep as is
            # If shape is [batch, num_stack, num_obs], transpose
            # Check: if dim2 (last dimension) matches num_obs, then we have [batch, num_stack, num_obs] -> need transpose
            if stacked_obs.shape[-1] == self.num_obs:
                # Currently [batch, num_stack, num_obs] -> transpose to [batch, num_obs, num_stack]
                stacked_obs = stacked_obs.transpose(-2, -1)

            embedding = self.adaptation_module(stacked_obs)

            # Reshape embedding back to 3D if input was 4D: [H*N, emb] -> [H, N, emb]
            if original_shape is not None:
                embedding = embedding.view(original_shape[0], original_shape[1], -1)
        act_input = torch.cat((obs, embedding), dim=-1)
        action_mean = self.actor(act_input)
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        return dist, embedding
    
    def est_value(self, obs, privileged_obs):
        critic_input = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)
    
    def ac_parameters(self):
        for p in self.critic.parameters():
            yield p
        for p in self.actor.parameters():
            yield p
        for p in self.privileged_encoder.parameters():
            yield p
        yield self.logstd

    def adapt_parameters(self):
        for p in self.adaptation_module.parameters():
            yield p
