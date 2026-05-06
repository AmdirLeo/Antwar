from __future__ import annotations

import sys
import time
import traceback
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


def debug_log(msg: str):
    """
    统一的调试日志输出，写入标准错误流 (stderr)。
    Saiblo 平台和本地测试脚本会将 stderr 记录下来，方便复盘分析。
    """
    print(f"[SmartGreedy] {msg}", file=sys.stderr, flush=True)


class SmartGreedyAgent(BaseAgent):
    """
    Advanced heuristic greedy agent.
    Sense -> Evaluate -> Act
    """

    def calculate_threat_level(self, state: BackendState, player: int) -> float:
        enemy_ants = [ant for ant in state.ants if ant.player == 1 - player and ant.hp > 0]
        if not enemy_ants:
            return 0.0

        base_x, base_y = PLAYER_BASES[player]
        threat_score = 0.0

        for ant in enemy_ants:
            dist = hex_distance(ant.x, ant.y, base_x, base_y)
            weight = 5.0 if ant.kind == AntKind.COMBAT else 1.0

            if dist < 10:
                threat_score += weight * (10 - dist) * 2.0
            elif dist < 20:
                threat_score += weight * (20 - dist) * 0.5

        return threat_score

    def _evaluate_clustering(self, state: BackendState, player: int) -> Tuple[float, int, int]:
        enemy_ants = [ant for ant in state.ants if ant.player == 1 - player and ant.hp > 0]
        if not enemy_ants:
            return (0.0, -1, -1)

        best_score = 0.0
        best_x = -1
        best_y = -1

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

    def choose_bundle(self, state: BackendState, player: int, bundles: Optional[List[ActionBundle]] = None) -> ActionBundle:
        start_time = time.time()
        
        # 为了防止评测系统因为我们代码某处抛出异常而直接判负，加上全局异常捕获
        try:
            if bundles is None:
                bundles = self.catalog.build(state, player, rerank=False)

            # 寻找真正的“无操作(No-op)” Bundle，以防 bundles[0] 是造塔动作
            noop_bundle = next((b for b in bundles if not b.operations), None)
            if not bundles or noop_bundle is None:
                # 极端后备方案
                return ActionBundle(name="hold", score=0.0, tags=("noop",))

            threat_level = self.calculate_threat_level(state, player)
            cluster_score, cluster_x, cluster_y = self._evaluate_clustering(state, player)
            coins = state.coins[player]

            debug_log(f"--- Round State --- | Coins: {coins} | Threat: {threat_level:.1f} | Max Enemy Cluster Score: {cluster_score:.1f} at ({cluster_x}, {cluster_y})")

            enemy_ants = [ant for ant in state.ants if ant.player == 1 - player and ant.hp > 0]
            my_towers = [t for t in state.towers if t.player == player]
            ice_tower_positions = [(t.x, t.y) for t in my_towers if t.tower_type == TowerType.ICE]

            best_bundle = noop_bundle
            best_score = -9999.0

            for bundle in bundles:
                custom_score = 0.0

                if not bundle.operations:
                    # 对于空操作，如果我们处于安全且没钱的状态，给点小奖励
                    if threat_level == 0 and coins < 200:
                        custom_score = 10.0
                    else:
                        custom_score = 0.0

                for op in bundle.operations:
                    # -- Super Weapons --
                    if op.op_type in (OperationType.USE_LIGHTNING_STORM, OperationType.USE_EMP_BLASTER):
                        if cluster_score > 15.0 and hex_distance(op.arg0, op.arg1, cluster_x, cluster_y) <= 2:
                            custom_score += 1000.0 + cluster_score * 10.0
                            debug_log(f"High value Super Weapon target found at ({op.arg0}, {op.arg1})!")

                    # -- Tower Cooperation (Build/Upgrade) --
                    elif op.op_type == OperationType.BUILD_TOWER:
                        min_dist_to_enemy = min([hex_distance(op.arg0, op.arg1, ant.x, ant.y) for ant in enemy_ants] + [999])
                        if 1 <= min_dist_to_enemy <= 4:
                            custom_score += 50.0 - min_dist_to_enemy * 5.0

                        # Proactive Strategy: Map Control & Outpost Expansion
                        if threat_level < 10.0 and coins > 40 and len(my_towers) < 3:
                            # Map Control Base Score
                            custom_score += 100.0

                            # Outpost score: closer to x=9 is better
                            dist_to_center_x = abs(op.arg0 - 9)
                            outpost_score = (9 - dist_to_center_x) * 5.0

                            # Further from base is better
                            base_x, base_y = PLAYER_BASES[player]
                            dist_to_base = hex_distance(op.arg0, op.arg1, base_x, base_y)
                            outpost_score += dist_to_base * 2.0

                            custom_score += outpost_score

                        for t in my_towers:
                            if hex_distance(op.arg0, op.arg1, t.x, t.y) <= 1:
                                custom_score -= 10.0

                    elif op.op_type == OperationType.UPGRADE_TOWER:
                        target_type = op.arg1
                        tower_id = op.arg0
                        target_tower = next((t for t in my_towers if t.tower_id == tower_id), None)

                        if target_tower:
                            # Exploding Combat Ant handling
                            near_exploding_ant = False
                            for ant in enemy_ants:
                                if ant.kind == AntKind.COMBAT and ant.hp < 10 and hex_distance(target_tower.x, target_tower.y, ant.x, ant.y) <= 2:
                                    near_exploding_ant = True
                                    break

                            if near_exploding_ant:
                                if target_type in (TowerType.HEAVY, TowerType.HEAVY_PLUS):
                                    custom_score += 200.0

                            if target_type in (TowerType.ICE, TowerType.HEAVY_PLUS, TowerType.MORTAR_PLUS):
                                custom_score += 60.0
                                if target_type in (TowerType.HEAVY_PLUS, TowerType.MORTAR_PLUS):
                                    for ix, iy in ice_tower_positions:
                                        if hex_distance(target_tower.x, target_tower.y, ix, iy) <= 3:
                                            custom_score += 40.0
                                if target_type == TowerType.ICE:
                                    for t in my_towers:
                                        if t.tower_type in (TowerType.HEAVY, TowerType.HEAVY_PLUS, TowerType.MORTAR, TowerType.MORTAR_PLUS):
                                            if hex_distance(target_tower.x, target_tower.y, t.x, t.y) <= 3:
                                                custom_score += 40.0

                            min_dist_to_enemy = min([hex_distance(target_tower.x, target_tower.y, ant.x, ant.y) for ant in enemy_ants] + [999])
                            if min_dist_to_enemy <= 5:
                                custom_score += 20.0

                    # -- Base Upgrade --
                    elif op.op_type in (OperationType.UPGRADE_GENERATION_SPEED, OperationType.UPGRADE_GENERATED_ANT):
                        if threat_level < 20.0 and coins > 300:
                            custom_score += 80.0
                        elif threat_level > 50.0:
                            custom_score -= 100.0

                    # -- Downgrade/Sell --
                    elif op.op_type == OperationType.DOWNGRADE_TOWER:
                        tower_id = op.arg0
                        target_tower = next((t for t in my_towers if t.tower_id == tower_id), None)
                        if target_tower:
                            # Exploding Combat Ant handling
                            near_exploding_ant = False
                            for ant in enemy_ants:
                                if ant.kind == AntKind.COMBAT and ant.hp < 10 and hex_distance(target_tower.x, target_tower.y, ant.x, ant.y) <= 2:
                                    near_exploding_ant = True
                                    break

                            if near_exploding_ant:
                                custom_score += 150.0 # salvage points before explosion destroys it

                            # 尝试安全地获取 max_hp，如果由于 SDK 版本不同获取不到，则保底为 15
                            max_hp = 15
                            if hasattr(TOWER_STATS, 'get'):
                                tower_stat = TOWER_STATS.get(target_tower.tower_type)
                                if tower_stat and hasattr(tower_stat, 'max_hp'):
                                    max_hp = tower_stat.max_hp
                                    
                            if target_tower.hp > 0 and target_tower.hp < max_hp * 0.2:
                                custom_score += 70.0
                                debug_log(f"Selling dying tower {tower_id} at ({target_tower.x}, {target_tower.y})")

                            min_dist_to_enemy = min([hex_distance(target_tower.x, target_tower.y, ant.x, ant.y) for ant in enemy_ants] + [999])
                            if min_dist_to_enemy > 12 and threat_level > 0:
                                custom_score += 40.0

                if custom_score > best_score:
                    best_score = custom_score
                    best_bundle = bundle

            # 如果得分最高的操作也毫无意义（<=0），强制选择空操作
            if best_score <= 0.0:
                best_bundle = noop_bundle

            elapsed_time = (time.time() - start_time) * 1000
            
            # 记录决策结果
            action_desc = best_bundle.name if best_bundle.name else "Wait/Pass"
            debug_log(f"Action Decided: {action_desc} | Score: {best_score:.1f} | Decision Time: {elapsed_time:.2f} ms")
            
            return best_bundle

        except Exception as e:
            # 当发生不可预知的崩溃时，记录异常栈并安全返回跳过动作
            debug_log(f"CRITICAL ERROR in choose_bundle: {str(e)}")
            traceback.print_exc(file=sys.stderr)
            return ActionBundle(name="hold_fallback", score=0.0, tags=("noop",))

class AI(SmartGreedyAgent):
    pass
