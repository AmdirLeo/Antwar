from __future__ import annotations

from dataclasses import dataclass

from SDK.utils.features import FeatureExtractor
from SDK.training.base import BaseSelfPlayTrainer, EpisodeBatch
from SDK.training.policies import MaskedLinearPolicy


@dataclass(slots=True)
class TrainerConfig:
    gamma: float = 0.99
    episodes_per_batch: int = 4
    learning_rate: float = 1e-2
    value_learning_rate: float = 5e-3
    seed: int = 0


class LinearSelfPlayTrainer(BaseSelfPlayTrainer):
    def __init__(self, env_factory, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        feature_extractor = FeatureExtractor()
        warmup_env = env_factory(seed=self.config.seed)
        observations, _ = warmup_env.reset(seed=self.config.seed)
        first_obs = observations["player_0"]
        obs_dim = len(feature_extractor.flatten_observation(first_obs))
        action_dim = len(first_obs["action_mask"])
        warmup_env.close()
        self.policy = MaskedLinearPolicy(obs_dim=obs_dim, action_dim=action_dim, seed=self.config.seed)
        super().__init__(
            env_factory=env_factory,
            gamma=self.config.gamma,
            episodes_per_batch=self.config.episodes_per_batch,
            seed=self.config.seed,
        )

    def select_action(self, observation, explore: bool = True) -> int:
        flat = self.feature_extractor.flatten_observation(observation)
        step = self.policy.step(flat, observation["action_mask"].astype(float), explore=explore)
        return step.action

    def update_from_batch(self, batch: EpisodeBatch) -> dict[str, float]:
        return self.policy.update(
            observations=batch.observations,
            masks=batch.masks,
            actions=batch.actions,
            returns=batch.returns,
            learning_rate=self.config.learning_rate,
            value_learning_rate=self.config.value_learning_rate,
        )
