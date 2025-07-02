from __future__ import annotations
import torch
import torch.nn as nn
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.common.torch_layers import create_mlp


class LSTMTD3Policy(TD3Policy):
    """
    TD3 policy whose **actor path** passes through an LSTM.
    Critics remain the default feed-forward nets.
    """

    def __init__(self, observation_space, action_space, lr_schedule, lstm_hidden: int = 128, **kwargs):
        # Let parent build everything first (features_dim will be known *after* _setup_model)
        self._custom_lstm_hidden = lstm_hidden
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    # --------------------------------------------------------------------- #
    #  Override: called during model.setup() – features_dim is available.
    # --------------------------------------------------------------------- #
    def _build_mlp_extractor(self) -> None:
        # ----- shared feature extractor (MLP) -----
        mlp_hidden = 256
        self.mlp_extractor = nn.Sequential(
            *create_mlp(self.features_dim, [mlp_hidden], mlp_hidden, activation_fn=nn.ReLU)
        )

        # ----- LSTM on top (actor only) ----------
        self.lstm = nn.LSTM(
            input_size=mlp_hidden,
            hidden_size=self._custom_lstm_hidden,
            batch_first=True,
        )
        self.actor_mu = nn.Linear(self._custom_lstm_hidden, self.action_space.shape[0])

        # Critic networks → keep default two‐headed MLPs created by parent
        # (they will re-use self.mlp_extractor)

    # --------------------------------------------------------------------- #
    #  Actor forward (used by TD3Policy.predict())
    # --------------------------------------------------------------------- #
    def forward(self, obs, deterministic=True):
        batch = obs.shape[0]
        features = self.extract_features(obs)            # base extractor
        features = self.mlp_extractor(features).unsqueeze(1)  # add seq length = 1
        lstm_out, _ = self.lstm(features)
        last = lstm_out[:, -1, :]                        # (batch, hidden)
        mean_actions = torch.tanh(self.actor_mu(last))
        return mean_actions 