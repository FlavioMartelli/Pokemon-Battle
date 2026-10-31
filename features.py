"""Feature extraction for the Pokemon battle outcome task.

Each battle becomes one row of hand-built features:
  - static team features (base stats, levels, items, type matchups)
  - battle state at turn 30 (HP left, faints, status, boosts, active matchup)
  - trend features over the timeline (HP difference / damage / KOs per 10-turn batch)
  - move-based features (estimated damage, physical vs special usage, effectiveness)
  - field tracking (spikes per side, key moves like explosion/curse/rest, peak boosts)
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
BAD_STATUS = ["slp", "par", "tox", "psn", "frz", "brn"]
EFFECTS = ["substitute", "leechseed", "confusion", "trapped", "perish"]
KEY_MOVES = ["spikes", "rapidspin", "explosion", "selfdestruct", "curse", "bellydrum", "swordsdance",
             "growth", "rest", "sleeptalk", "sleeppowder", "lovelykiss", "hypnosis", "spore",
             "thunderwave", "toxic", "roar", "whirlwind"]
CHECKPOINTS = [10, 20, 30]
LOW_HP = 0.25


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_type_chart(path):
    chart = defaultdict(lambda: 1.0)
    for row in pd.read_csv(path).itertuples(index=False):
        chart[(row.attacking_type, row.defending_type)] = float(row.multiplier)
    return chart


def build_pokedex(battles):
    """Name -> {types, base stats} from every Pokemon whose stats we see (p1 teams, p2 leads)."""
    dex = {}
    for b in battles:
        for p in b["p1_team_details"] + [b["p2_lead_details"]]:
            dex[p["name"]] = {"types": [t for t in p["types"] if t != "notype"], **{k: p[k] for k in STATS}}
    # Fallback for Pokemon only ever seen in p2 timelines: average stats, no types.
    mean = {k: float(np.mean([v[k] for v in dex.values()])) for k in STATS}
    dex["__unknown__"] = {"types": [], **mean}
    return dex


class FeatureBuilder:
    def __init__(self, dex, chart):
        self.dex = dex
        self.chart = chart

    def info(self, name):
        return self.dex.get(name, self.dex["__unknown__"])

    def eff(self, move_type, defender_types):
        m = 1.0
        for t in defender_types:
            m *= self.chart[(move_type, t)]
        return m

    def best_stab_eff(self, att_types, def_types):
        """Best type multiplier an attacker gets on its own types against a defender."""
        if not att_types:
            return 1.0
        return max(self.eff(t, def_types) for t in att_types)

    def matchup(self, side_a, side_b):
        """Mean over (a, b) pairs of a's best STAB multiplier on b minus b's on a."""
        if not side_a or not side_b:
            return 0.0
        vals = [
            np.log2(max(self.best_stab_eff(self.info(a)["types"], self.info(b)["types"]), 0.25))
            - np.log2(max(self.best_stab_eff(self.info(b)["types"], self.info(a)["types"]), 0.25))
            for a in side_a for b in side_b
        ]
        return float(np.mean(vals))

    def move_damage(self, move, attacker, defender):
        """Rough damage estimate: power * STAB * effectiveness * attack/defense ratio * accuracy."""
        if not move or move["base_power"] == 0 or move["category"] == "status":
            return 0.0, 1.0
        a, d = self.info(attacker), self.info(defender)
        if move["category"] == "physical":
            ratio = a["base_atk"] / d["base_def"]
        else:
            ratio = a["base_spa"] / d["base_spd"]
        stab = 1.5 if move["type"] in a["types"] else 1.0
        e = self.eff(move["type"], d["types"])
        return move["base_power"] * stab * e * ratio * move["accuracy"], e

    def build(self, b):
        f = {}
        team1 = b["p1_team_details"]
        tl = b["battle_timeline"]
        names1 = [p["name"] for p in team1]

        # ---- static team features -------------------------------------------------
        p1_stats = np.array([[p[k] for k in STATS] for p in team1], dtype=float)
        p2_names = list(dict.fromkeys(t["p2_pokemon_state"]["name"] for t in tl))
        p2_stats = np.array([[self.info(n)[k] for k in STATS] for n in p2_names], dtype=float)
        p1_mean, p2_mean = p1_stats.mean(0), p2_stats.mean(0)
        for i, k in enumerate(STATS):
            s = k[5:]
            f[f"p1_mean_{s}"] = p1_mean[i]
            f[f"p2_mean_{s}"] = p2_mean[i]
            f[f"diff_mean_{s}"] = p1_mean[i] - p2_mean[i]
        f["p1_atk_minus_p2_def"] = p1_mean[1] - p2_mean[2]
        f["p1_spa_minus_p2_spd"] = p1_mean[3] - p2_mean[4]
        f["p2_atk_minus_p1_def"] = p2_mean[1] - p1_mean[2]
        f["p2_spa_minus_p1_spd"] = p2_mean[3] - p1_mean[4]
        f["p1_mean_level"] = np.mean([p["level"] for p in team1])
        f["p1_n_leftovers"] = sum(p["item"] == "leftovers" for p in team1)
        f["p2_revealed"] = len(p2_names)
        f["type_matchup_all"] = self.matchup(names1, p2_names)
        lead2 = b["p2_lead_details"]["name"]
        f["lead_matchup"] = self.matchup([tl[0]["p1_pokemon_state"]["name"]], [lead2])

        # ---- walk the timeline ----------------------------------------------------
        hp1 = {n: 1.0 for n in names1}
        hp2 = {}
        status1, status2 = {}, {}
        fainted1, fainted2 = set(), set()
        prev_active1 = prev_active2 = None
        dmg_dealt = np.zeros(len(CHECKPOINTS))
        dmg_taken = np.zeros(len(CHECKPOINTS))
        ko_by_batch1 = np.zeros(len(CHECKPOINTS))  # p1 Pokemon lost
        ko_by_batch2 = np.zeros(len(CHECKPOINTS))  # p2 Pokemon lost
        snap = {}
        mv = defaultdict(float)
        last_ko1 = last_ko2 = 0
        # Spikes on each side of the field (Gen 2: one layer, removed by the victim's Rapid Spin).
        spikes_on = {"p1": False, "p2": False}
        rested = {"p1": set(), "p2": set()}
        max_boost = {"p1": 0, "p2": 0}

        for t in tl:
            turn = t["turn"]
            batch = min((turn - 1) * len(CHECKPOINTS) // 30, len(CHECKPOINTS) - 1)
            s1, s2 = t["p1_pokemon_state"], t["p2_pokemon_state"]
            m1, m2 = t["p1_move_details"], t["p2_move_details"]
            n1, n2 = s1["name"], s2["name"]

            # Damage = HP lost since we last saw that Pokemon (positive), heals are ignored.
            dmg_taken[batch] += max(hp1.get(n1, 1.0) - s1["hp_pct"], 0.0)
            dmg_dealt[batch] += max(hp2.get(n2, 1.0) - s2["hp_pct"], 0.0)
            hp1[n1], hp2[n2] = s1["hp_pct"], s2["hp_pct"]
            status1[n1], status2[n2] = s1["status"], s2["status"]
            if s1["status"] == "fnt" and n1 not in fainted1:
                fainted1.add(n1); ko_by_batch1[batch] += 1; last_ko1 = turn
            if s2["status"] == "fnt" and n2 not in fainted2:
                fainted2.add(n2); ko_by_batch2[batch] += 1; last_ko2 = turn

            # Null move: a switch if the active Pokemon changed, otherwise it could not act.
            for side, other, m, s, prev in (("p1", "p2", m1, s1, prev_active1),
                                            ("p2", "p1", m2, s2, prev_active2)):
                n = s["name"]
                if m is None:
                    if n != prev:
                        mv[f"{side}_switches"] += 1
                        if spikes_on[side] and turn > 1:
                            mv[f"{side}_switches_into_spikes"] += 1
                    else:
                        mv[f"{side}_cant_move"] += 1
                else:
                    mv[f"{side}_{m['category']}_moves"] += 1
                    if m["priority"] > 0:
                        mv[f"{side}_priority_moves"] += 1
                    if m["name"] in KEY_MOVES:
                        mv[f"{side}_used_{m['name']}"] += 1
                    if m["name"] == "spikes":
                        spikes_on[other] = True
                    elif m["name"] == "rapidspin":
                        spikes_on[side] = False
                    elif m["name"] == "rest":
                        rested[side].add(n)
                max_boost[side] = max(max_boost[side], sum(s["boosts"].values()))
            mv["p1_spikes_turns"] += spikes_on["p1"]
            mv["p2_spikes_turns"] += spikes_on["p2"]
            d1, e1 = self.move_damage(m1, n1, n2)
            d2, e2 = self.move_damage(m2, n2, n1)
            mv["p1_est_dmg"] += d1
            mv["p2_est_dmg"] += d2
            if d1 > 0:
                mv["p1_attacks"] += 1; mv["p1_eff_sum"] += np.log2(max(e1, 0.25))
                mv["p1_super_eff"] += e1 > 1
            if d2 > 0:
                mv["p2_attacks"] += 1; mv["p2_eff_sum"] += np.log2(max(e2, 0.25))
                mv["p2_super_eff"] += e2 > 1
            prev_active1, prev_active2 = n1, n2

            if turn in CHECKPOINTS:
                # Unseen p2 Pokemon assumed at full HP (6 per team).
                hp_left1 = sum(hp1.values())
                hp_left2 = sum(hp2.values()) + (6 - len(hp2))
                snap[turn] = (hp_left1, hp_left2, len(fainted1), len(fainted2))

        # ---- state at the last checkpoint + trends ---------------------------------
        for c in CHECKPOINTS:
            h1, h2, k1, k2 = snap[c]
            f[f"hp_diff_t{c}"] = h1 - h2
            f[f"faint_diff_t{c}"] = k2 - k1
        h1, h2, k1, k2 = snap[30]
        f["p1_hp_left"], f["p2_hp_left"] = h1, h2
        f["p1_fainted"], f["p2_fainted"] = k1, k2
        f["hp_diff_change_last10"] = f["hp_diff_t30"] - f["hp_diff_t20"]
        f["hp_diff_change_first10"] = f["hp_diff_t10"]
        for i, c in enumerate(CHECKPOINTS):
            f[f"dmg_dealt_b{i}"] = dmg_dealt[i]
            f[f"dmg_taken_b{i}"] = dmg_taken[i]
            f[f"dmg_net_b{i}"] = dmg_dealt[i] - dmg_taken[i]
            f[f"ko_net_b{i}"] = ko_by_batch2[i] - ko_by_batch1[i]
        f["last_ko_p1_lost"], f["last_ko_p2_lost"] = last_ko1, last_ko2
        f["last_ko_by_p1"] = int(last_ko2 > last_ko1)

        alive1 = [n for n in names1 if n not in fainted1]
        alive2 = [n for n in p2_names if n not in fainted2]
        f["p1_alive"], f["p2_alive_seen"] = len(alive1), len(alive2)
        f["p2_alive_est"] = 6 - len(fainted2)
        f["p1_low_hp"] = sum(0 < hp1[n] < LOW_HP for n in alive1)
        f["p2_low_hp"] = sum(0 < hp2[n] < LOW_HP for n in alive2)
        f["p1_alive_hp_mean"] = np.mean([hp1[n] for n in alive1]) if alive1 else 0.0
        f["p2_alive_hp_mean"] = np.mean([hp2[n] for n in alive2]) if alive2 else 0.0
        f["type_matchup_alive"] = self.matchup(alive1, alive2)
        for st in BAD_STATUS:
            a = sum(status1.get(n) == st for n in alive1)
            c = sum(status2.get(n) == st for n in alive2)
            f[f"p1_{st}"], f[f"p2_{st}"], f[f"diff_{st}"] = a, c, c - a
        f["p1_statused"] = sum(status1.get(n) in BAD_STATUS for n in alive1)
        f["p2_statused"] = sum(status2.get(n) in BAD_STATUS for n in alive2)

        # Active matchup at turn 30.
        last = tl[-1]
        a1, a2 = last["p1_pokemon_state"], last["p2_pokemon_state"]
        f["active_hp_diff"] = a1["hp_pct"] - a2["hp_pct"]
        f["active_matchup"] = self.matchup([a1["name"]], [a2["name"]])
        f["active_speed_diff"] = self.info(a1["name"])["base_spe"] - self.info(a2["name"])["base_spe"]
        f["p1_active_boosts"] = sum(a1["boosts"].values())
        f["p2_active_boosts"] = sum(a2["boosts"].values())
        f["active_boost_diff"] = f["p1_active_boosts"] - f["p2_active_boosts"]
        for e in EFFECTS:
            f[f"p1_eff_{e}"] = int(e in a1["effects"])
            f[f"p2_eff_{e}"] = int(e in a2["effects"])

        # Move usage.
        for side in ("p1", "p2"):
            for k in ("switches", "cant_move", "physical_moves", "special_moves", "status_moves",
                      "priority_moves", "est_dmg", "super_eff"):
                f[f"{side}_{k}"] = mv[f"{side}_{k}"]
            n_att = mv[f"{side}_attacks"]
            f[f"{side}_mean_eff"] = mv[f"{side}_eff_sum"] / n_att if n_att else 0.0
            f[f"{side}_mean_est_dmg"] = mv[f"{side}_est_dmg"] / n_att if n_att else 0.0
            phys, spec = mv[f"{side}_physical_moves"], mv[f"{side}_special_moves"]
            f[f"{side}_special_frac"] = spec / (phys + spec) if phys + spec else 0.5
        for side in ("p1", "p2"):
            for k in KEY_MOVES:
                f[f"{side}_used_{k}"] = mv[f"{side}_used_{k}"]
            f[f"{side}_spikes_on_side"] = int(spikes_on[side])
            f[f"{side}_spikes_turns"] = mv[f"{side}_spikes_turns"]
            f[f"{side}_switches_into_spikes"] = mv[f"{side}_switches_into_spikes"]
            f[f"{side}_max_boost"] = max_boost[side]
        f["spikes_diff"] = f["p2_spikes_on_side"] - f["p1_spikes_on_side"]
        f["p1_rested_alive"] = len(rested["p1"] & set(alive1))
        f["p2_rested_alive"] = len(rested["p2"] & set(alive2))
        f["est_dmg_diff"] = f["p1_est_dmg"] - f["p2_est_dmg"]
        f["mean_eff_diff"] = f["p1_mean_eff"] - f["p2_mean_eff"]
        f["cant_move_diff"] = f["p2_cant_move"] - f["p1_cant_move"]
        return f


def make_features(train, test, type_chart_path):
    dex = build_pokedex(train + test)
    fb = FeatureBuilder(dex, load_type_chart(type_chart_path))
    X_train = pd.DataFrame([fb.build(b) for b in train])
    X_test = pd.DataFrame([fb.build(b) for b in test])
    return X_train, X_test
