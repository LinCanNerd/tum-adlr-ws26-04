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
     
        )
        self.adaptation_module = torch.nn.Sequential(
            torch.nn.Linear(num_obs + obs_stacking, 1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, num_embedding),
        )
        self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_act), fill_value=-2.0), requires_grad=True)

        def act(self, obs, privileged_obs = None, stacked_obs = None):
        print("===== USING THIS ACT FUNCTION =====")

        # 强制 obs 变成 2D
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        else:
            obs = obs.view(1, -1)

        # 得到 embedding
        if privileged_obs is not None:
            embedding = self.privileged_encoder(privileged_obs)
        elif stacked_obs is not None:
            embedding = self.adaptation_module(stacked_obs.flatten(start_dim=-2))
        else:
            raise ValueError("Neither privileged_obs nor stacked_obs provided")

        # 强制 embedding 变成 2D
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)
        else:
            embedding = embedding.view(1, -1)

        # 打印确认形状
        print("DEBUG SHAPES:", obs.shape, embedding.shape)

        # 拼接输入
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
