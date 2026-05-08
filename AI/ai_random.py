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
    print(f"[log] {msg}", file=sys.stderr, flush=True)


class AdvancedSearchAgent(BaseAgent):
    """
    Advanced Search Agent with Time Control and Advantage Lookahead.
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

    def evaluate_bundle_heuristic(self, bundle: ActionBundle, state: BackendState, player: int, threat_level: float, cluster_score: float, cluster_x: int, cluster_y: int, coins: int, enemy_ants: list, my_towers: list, ice_tower_positions: list) -> float:
        custom_score = 0.0
        is_endgame = state.round_index > 480

        if not bundle.operations:
            if threat_level == 0 and coins < 200:
                custom_score = 10.0
            else:
                custom_score = 0.0

        is_saving_for_super_weapon = (cluster_score >= 8.0 and coins < 100)

        for op in bundle.operations:
            if is_saving_for_super_weapon and op.op_type in (OperationType.BUILD_TOWER, OperationType.UPGRADE_TOWER, OperationType.UPGRADE_GENERATION_SPEED, OperationType.UPGRADE_GENERATED_ANT):
                custom_score -= 10000.0

            # -- Super Weapons --
            if op.op_type in (OperationType.USE_LIGHTNING_STORM, OperationType.USE_EMP_BLASTER):
                if is_endgame:
                    custom_score -= 5000.0
                elif cluster_score > 15.0 and hex_distance(op.arg0, op.arg1, cluster_x, cluster_y) <= 2:
                    custom_score += 1000.0 + cluster_score * 10.0

            # -- Tower Cooperation (Build/Upgrade) --
            elif op.op_type == OperationType.BUILD_TOWER:
                min_dist_to_enemy = min([hex_distance(op.arg0, op.arg1, ant.x, ant.y) for ant in enemy_ants] + [999])
                if 1 <= min_dist_to_enemy <= 4:
                    custom_score += 50.0 - min_dist_to_enemy * 5.0

                if threat_level < 10.0 and coins > 40 and len(my_towers) < 3:
                    custom_score += 100.0
                    dist_to_center_x = abs(op.arg0 - 9)
                    outpost_score = (9 - dist_to_center_x) * 5.0
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

                if is_endgame:
                    custom_score += 300.0

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
                    near_exploding_ant = False
                    for ant in enemy_ants:
                        if ant.kind == AntKind.COMBAT and ant.hp < 10 and hex_distance(target_tower.x, target_tower.y, ant.x, ant.y) <= 2:
                            near_exploding_ant = True
                            break

                    if near_exploding_ant:
                        custom_score += 150.0

                    max_hp = 15
                    if hasattr(TOWER_STATS, 'get'):
                        tower_stat = TOWER_STATS.get(target_tower.tower_type)
                        if tower_stat and hasattr(tower_stat, 'max_hp'):
                            max_hp = tower_stat.max_hp

                    if target_tower.hp > 0 and target_tower.hp < max_hp * 0.2:
                        custom_score += 70.0

                    min_dist_to_enemy = min([hex_distance(target_tower.x, target_tower.y, ant.x, ant.y) for ant in enemy_ants] + [999])
                    if min_dist_to_enemy > 12 and threat_level > 0:
                        custom_score += 40.0

        return custom_score

    def evaluate_leaf_state(self, state: BackendState, player: int) -> float:
        """
        高级非线性叶子节点评估 (Advanced Non-linear Leaf Evaluation)
        """
        my_hp = state.bases[player].hp
        enemy_hp = state.bases[1 - player].hp
        my_towers = [t for t in state.towers if t.player == player]
        threat = self.calculate_threat_level(state, player)
        coins = state.coins[player]
        round_index = state.round_index
        
        # 1. 终极生死特判
        if my_hp <= 0:
            return -999999.0  # 死局，绝对否定
        if enemy_hp <= 0:
            return 999999.0   # 斩杀，绝对肯定

        score = 0.0

        # 2. 生命值：二次方风险厌恶曲线 (满血 50)
        # my_hp / 50.0 是 0~1 的比例。(比例)^2 会让高血量时变化平缓，低血量时断崖式下跌
        # 这里用正数加分，意味着从 50 掉到 40 扣分少，从 20 掉到 10 扣分极大
        score += (my_hp / 50.0) ** 2 * 5000.0
        score -= (enemy_hp / 50.0) ** 2 * 2000.0  # 压低敌方血量的奖励

        # 大后期（平局判定期）绝对保血量
        if round_index > 480:
            score += my_hp * 10000.0

        # 3. 固定资产净值计算 (Net Worth)
        tower_value = 0.0
        for t in my_towers:
            base_val = 30.0  # 假定基础塔造价
            upgrade_multiplier = 1.0
            
            # 识别高级塔给予倍率奖励
            tower_type_str = str(t.tower_type)
            if "PLUS" in tower_type_str:
                upgrade_multiplier = 2.5
            elif "HEAVY" in tower_type_str or "MORTAR" in tower_type_str or "ICE" in tower_type_str:
                upgrade_multiplier = 1.8
                
            # 如果塔快死了，价值折损
            hp_ratio = t.hp / 15.0 if t.hp < 15.0 else 1.0
            
            tower_value += (base_val * upgrade_multiplier) * hp_ratio

        # 赋予固定资产 1.3 倍的溢价，彻底根除“卖塔套现”的恶习
        score += tower_value * 1.3

        # 4. 现金流：边际效用递减机制
        if coins >= 135:
            # 超过超级武器所需金币后，钱的价值大打折扣，逼迫 AI 将钱转化为塔
            score += 135.0 * 1.0 + (coins - 135.0) * 0.2
        elif coins >= 90:
            # 凑齐闪电风暴，正常计价
            score += coins * 1.0
        else:
            # 极度缺钱时（放不出技能），钱稍微贵重一点点，鼓励适当储蓄
            score += coins * 1.1

        # 5. 动态威胁惩罚
        # 威胁值本身已经是考虑了距离的（10格以内翻倍），这里再乘一个系数
        # 随回合数增加，敌方波数变强，稍微降低对威胁的恐慌，避免满场跑不发育
        threat_multiplier = 15.0 if round_index < 200 else 10.0
        score -= threat * threat_multiplier

        return score

    def choose_bundle(self, state: BackendState, player: int, bundles: Optional[List[ActionBundle]] = None) -> ActionBundle:
        start_time = time.time()

        try:
            if bundles is None:
                bundles = self.catalog.build(state, player, rerank=False)

            noop_bundle = next((b for b in bundles if not b.operations), None)
            if not bundles or noop_bundle is None:
                return ActionBundle(name="hold", score=0.0, tags=("noop",))

            threat_level = self.calculate_threat_level(state, player)
            cluster_score, cluster_x, cluster_y = self._evaluate_clustering(state, player)
            coins = state.coins[player]

            round_log_msg = f"--- Round State --- | Coins: {coins} | Threat: {threat_level:.1f} | Max Enemy Cluster Score: {cluster_score:.1f} at ({cluster_x}, {cluster_y})"

            enemy_ants = [ant for ant in state.ants if ant.player == 1 - player and ant.hp > 0]
            my_towers = [t for t in state.towers if t.player == player]
            ice_tower_positions = [(t.x, t.y) for t in my_towers if t.tower_type == TowerType.ICE]

            # 1. Action Space Pruning
            scored_bundles = []
            for bundle in bundles:
                score = self.evaluate_bundle_heuristic(bundle, state, player, threat_level, cluster_score, cluster_x, cluster_y, coins, enemy_ants, my_towers, ice_tower_positions)
                scored_bundles.append((score, bundle))

            scored_bundles.sort(key=lambda x: x[0], reverse=True)
            K = min(8, len(scored_bundles))
            top_k_bundles = scored_bundles[:K]

            best_overall_bundle = top_k_bundles[0][1] if top_k_bundles else noop_bundle
            best_overall_score = -99999.0

            # 2. Iterative Deepening Lookahead Search with Advantage Evaluation
            depth = 1
            while time.time() - start_time < 9.0:
                depth_best_bundle = noop_bundle
                depth_best_score = -99999.0
                timeout_occurred = False

                # --- Calculate Baseline (NOOP) for Advantage ---
                noop_leaf_score = 0.0
                try:
                    noop_sim = state.clone()
                    for step in range(depth):
                        if player == 0:
                            noop_sim.resolve_turn([], [])
                        else:
                            noop_sim.resolve_turn([], [])
                    noop_leaf_score = self.evaluate_leaf_state(noop_sim, player)
                except Exception:
                    noop_leaf_score = 0.0

                for base_score, bundle in top_k_bundles:
                    if time.time() - start_time > 9.0:
                        timeout_occurred = True
                        break

                    bundle_leaf_score = 0.0
                    try:
                        sim_state = state.clone()
                        for step in range(depth):
                            ops = bundle.operations if step == 0 else []
                            if player == 0:
                                sim_state.resolve_turn(ops, [])
                            else:
                                sim_state.resolve_turn([], ops)
                        bundle_leaf_score = self.evaluate_leaf_state(sim_state, player)
                    except Exception:
                        bundle_leaf_score = noop_leaf_score

                    # The true value is the heuristic base + the marginal advantage over doing nothing
                    advantage = bundle_leaf_score - noop_leaf_score
                    total_score = base_score + advantage

                    if total_score > depth_best_score:
                        depth_best_score = total_score
                        depth_best_bundle = bundle

                if not timeout_occurred:
                    best_overall_bundle = depth_best_bundle
                    best_overall_score = depth_best_score
                    depth += 1
                else:
                    break

            elapsed_time = (time.time() - start_time) * 1000
            action_desc = best_overall_bundle.name if best_overall_bundle.name else "Wait/Pass"
            is_noop = best_overall_bundle.name in ("hold", "noop", "Wait/Pass")

            # 仅当采取了实质性动作，或每 20 回合的“心跳”时输出日志，防止截断
            if (not is_noop) or (state.round_index % 20 == 0):
                debug_log(round_log_msg)
                debug_log(f"Action Decided: {action_desc} | Score: {best_overall_score:.1f} | Completed Depth: {depth-1} | Decision Time: {elapsed_time:.2f} ms")

            return best_overall_bundle

        except Exception as e:
            debug_log(f"CRITICAL ERROR in choose_bundle: {str(e)}")
            traceback.print_exc(file=sys.stderr)
            return ActionBundle(name="hold_fallback", score=0.0, tags=("noop",))

class AI(AdvancedSearchAgent):
    pass