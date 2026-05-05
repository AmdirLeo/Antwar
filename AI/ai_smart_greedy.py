from __future__ import annotations

import random
from typing import List, Tuple, Dict, Optional

try:
    from common import BaseAgent
except ModuleNotFoundError as exc:
    if exc.name != "common":
        raise
    from AI.common import BaseAgent

from SDK.utils.actions import ActionBundle
from SDK.utils.constants import OperationType, TowerType, SuperWeaponType, AntKind, PLAYER_BASES, TOWER_STATS
from SDK.backend.state import BackendState
from SDK.backend.model import Ant, Tower
from SDK.utils.geometry import hex_distance


class SmartGreedyAgent(BaseAgent):
    """
    Advanced heuristic greedy agent.
    Sense -> Evaluate -> Act
    """

    def calculate_threat_level(self, state: BackendState, player: int) -> float:
        """
        Calculate threat score based on enemy ants proximity to our base and their clustering.
        """
        enemy_ants = [ant for ant in state.ants if ant.player == 1 - player and ant.hp > 0]
        if not enemy_ants:
            return 0.0

        base_coords = PLAYER_BASES[player][0] # Assuming one base per player for simplicity
        threat_score = 0.0

        for ant in enemy_ants:
            # Distance to our base
            dist = hex_distance(ant.x, ant.y, base_coords[0], base_coords[1])

            # Weight by ant kind (Combat ants are much more dangerous)
            weight = 5.0 if ant.kind == AntKind.COMBAT else 1.0

            # Closer to base -> higher threat
            if dist < 10:
                threat_score += weight * (10 - dist) * 2.0
            elif dist < 20:
                threat_score += weight * (20 - dist) * 0.5

        return threat_score

    def _evaluate_clustering(self, state: BackendState, player: int) -> Tuple[float, int, int]:
        """
        Evaluate where the highest clustering of enemy ants is.
        Returns: (max_cluster_score, center_x, center_y)
        """
        enemy_ants = [ant for ant in state.ants if ant.player == 1 - player and ant.hp > 0]
        if not enemy_ants:
            return (0.0, -1, -1)

        best_score = 0.0
        best_x = -1
        best_y = -1

        # Consider each enemy ant as a potential cluster center
        for center_ant in enemy_ants:
            cluster_score = 0.0
            for ant in enemy_ants:
                if hex_distance(center_ant.x, center_ant.y, ant.x, ant.y) <= 3:
                    cluster_score += (5.0 if ant.kind == AntKind.COMBAT else 1.0)

            if cluster_score > best_score:
                best_score = cluster_score
                best_x = center_ant.x
                best_y = center_ant.y

        return (best_score, best_x, best_y)

    def _get_tower_coverage(self, state: BackendState, player: int) -> Dict[Tuple[int, int], int]:
        """
        Returns a dictionary mapping grid coordinates to the number of friendly towers covering it.
        """
        coverage: Dict[Tuple[int, int], int] = {}
        my_towers = [t for t in state.towers if t.player == player]

        # Simplified coverage map: just map the tower locations and their immediate ranges
        # To be strict on performance, we only evaluate coverage roughly or when needed.
        # But for scoring, we just check distance to existing towers.
        return coverage

    def choose_bundle(self, state: BackendState, player: int, bundles: Optional[List[ActionBundle]] = None) -> ActionBundle:
        """
        Evaluate and score each bundle, then return the highest scoring one.
        """
        # 1. State Perception
        if bundles is None:
            bundles = self.catalog.build(state, player, rerank=False)

        if not bundles:
            return ActionBundle(name="hold", score=0.0, tags=("noop",))

        threat_level = self.calculate_threat_level(state, player)
        cluster_score, cluster_x, cluster_y = self._evaluate_clustering(state, player)
        coins = state.coins[player]

        enemy_ants = [ant for ant in state.ants if ant.player == 1 - player and ant.hp > 0]
        my_towers = [t for t in state.towers if t.player == player]
        ice_tower_positions = [(t.x, t.y) for t in my_towers if t.tower_type == TowerType.ICE]

        # We will mutate a copy of scores
        best_bundle = bundles[0]
        best_score = -9999.0

        for bundle in bundles:
            custom_score = 0.0

            if not bundle.operations:
                # Noop bundle
                if threat_level == 0 and coins < 200:
                    custom_score = 10.0 # Small incentive to save money if safe
                else:
                    custom_score = 0.0

            for op in bundle.operations:
                # -- Super Weapons --
                if op.op_type in (OperationType.USE_LIGHTNING_STORM, OperationType.USE_EMP_BLASTER):
                    if cluster_score > 15.0 and hex_distance(op.arg0, op.arg1, cluster_x, cluster_y) <= 2:
                        custom_score += 1000.0 + cluster_score * 10.0 # Extremely high priority

                # -- Tower Cooperation (Build/Upgrade) --
                elif op.op_type == OperationType.BUILD_TOWER:
                    # Find distance to closest enemy ant
                    min_dist_to_enemy = 999
                    for ant in enemy_ants:
                        d = hex_distance(op.arg0, op.arg1, ant.x, ant.y)
                        if d < min_dist_to_enemy:
                            min_dist_to_enemy = d

                    if 1 <= min_dist_to_enemy <= 4:
                        custom_score += 50.0 - min_dist_to_enemy * 5.0 # Prefer 1-3 hexes

                    # Slightly penalize building too close to our own towers (spread out)
                    for t in my_towers:
                        if hex_distance(op.arg0, op.arg1, t.x, t.y) <= 1:
                            custom_score -= 10.0

                elif op.op_type == OperationType.UPGRADE_TOWER:
                    target_type = op.arg1
                    tower_id = op.arg0
                    target_tower = next((t for t in my_towers if t.tower_id == tower_id), None)

                    if target_tower:
                        if target_type in (TowerType.ICE, TowerType.HEAVY_PLUS, TowerType.MORTAR_PLUS):
                            custom_score += 60.0

                            # Synergy: Ice + High Damage
                            if target_type in (TowerType.HEAVY_PLUS, TowerType.MORTAR_PLUS):
                                # Is there an ice tower nearby?
                                for ix, iy in ice_tower_positions:
                                    if hex_distance(target_tower.x, target_tower.y, ix, iy) <= 3:
                                        custom_score += 40.0 # Great synergy

                            if target_type == TowerType.ICE:
                                # Is there a high damage tower nearby?
                                for t in my_towers:
                                    if t.tower_type in (TowerType.HEAVY, TowerType.HEAVY_PLUS, TowerType.MORTAR, TowerType.MORTAR_PLUS):
                                        if hex_distance(target_tower.x, target_tower.y, t.x, t.y) <= 3:
                                            custom_score += 40.0

                        # General upgrade value based on threat proximity
                        min_dist_to_enemy = min([hex_distance(target_tower.x, target_tower.y, ant.x, ant.y) for ant in enemy_ants] + [999])
                        if min_dist_to_enemy <= 5:
                            custom_score += 20.0

                # -- Base Upgrade --
                elif op.op_type in (OperationType.UPGRADE_GENERATION_SPEED, OperationType.UPGRADE_GENERATED_ANT):
                    if threat_level < 20.0 and coins > 300:
                        custom_score += 80.0 # Good long term investment
                    elif threat_level > 50.0:
                        custom_score -= 100.0 # Don't upgrade base if under heavy attack!

                # -- Downgrade/Sell --
                elif op.op_type == OperationType.DOWNGRADE_TOWER:
                    tower_id = op.arg0
                    target_tower = next((t for t in my_towers if t.tower_id == tower_id), None)
                    if target_tower:
                        # Sell if almost dead
                        max_hp = TOWER_STATS[target_tower.tower_type].max_hp if target_tower.tower_type in TOWER_STATS else 15
                        if target_tower.hp > 0 and target_tower.hp < max_hp * 0.2:
                            custom_score += 70.0

                        # Sell if very far from action
                        min_dist_to_enemy = min([hex_distance(target_tower.x, target_tower.y, ant.x, ant.y) for ant in enemy_ants] + [999])
                        if min_dist_to_enemy > 12 and threat_level > 0:
                            custom_score += 40.0

            if custom_score > best_score:
                best_score = custom_score
                best_bundle = bundle

        # If the best non-noop action has a score <= 0, and we have a noop action, prefer noop
        if best_score <= 0.0 and len(bundles) > 0:
            return bundles[0] # Usually the hold/noop action

        return best_bundle

class AI(SmartGreedyAgent):
    pass
