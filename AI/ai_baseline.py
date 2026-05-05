from __future__ import annotations

import random

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.utils.actions import ActionBundle
from SDK.utils.constants import OperationType
from SDK.backend import BackendState

class BaselineAgent(BaseAgent):
    """
    V1 Baseline AI for the environment test loop.

    This agent adopts a simple greedy approach:
    1. It reads the current state and explicitly checks if it has enough coins
       to build a Basic defense tower (Type ID 0). The cost calculation is
       explicitly 15 * (2 ** current_tower_count).
    2. If it has enough coins, it generates valid operations via ActionCatalog
       (without reranking for performance), and filters specifically for
       BUILD_TOWER operations which strictly build Basic towers.
    3. It randomly selects one build action if available.
    4. If it cannot build a tower, it will return the 'hold' action.
    """

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None
    ) -> ActionBundle:
        # Generate legal actions via the official encapsulation.
        # Rerank is set to False to strictly keep decision time under 1s.
        if bundles is None:
            bundles = self.catalog.build(state, player, rerank=False)

        if not bundles:
            return ActionBundle(name="hold", score=0.0, tags=("noop",))

        # The first bundle returned by ActionCatalog.build is always the 'hold' action
        hold_bundle = bundles[0]

        # 1. Check if coins are sufficient to build the most basic defense tower (Type ID 0).
        # We calculate the build cost for a new tower: 15 * (2 ** i).
        tower_count = state.tower_count(player)
        build_cost = 15 * (2 ** tower_count)

        if state.coins[player] >= build_cost:
            # 2. Filter the pre-generated bundles to find specifically BUILD_TOWER operations.
            # A BUILD_TOWER operation with x, y constructs a Type ID 0 (Basic) tower.
            build_bundles = [
                b for b in bundles
                if len(b.operations) == 1 and b.operations[0].op_type == OperationType.BUILD_TOWER
            ]

            # 3. Randomly select a valid empty coordinate to build a Basic tower
            if build_bundles:
                return random.choice(build_bundles)

        # 4. If coins are insufficient or no empty valid coordinate exists, hold.
        return hold_bundle

class AI(BaselineAgent):
    pass
