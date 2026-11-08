"""Feature extraction for the Pokemon battle outcome task.

Each battle becomes one row of hand-built features:
  - static team features (base stats, levels, items, type matchups)
  - battle state at turn 30 (HP left, faints, status, boosts, active matchup)
  - trend features over the timeline (HP difference / damage / KOs per 10-turn batch)
  - move-based features (estimated damage, physical vs special usage, effectiveness)
  - field tracking (spikes per side, key moves like explosion/curse/rest, peak boosts)
  - threat features: how many hits each side needs to KO the other's survivors, using movesets
    learned from all battles; p2's unseen Pokemon are filled in when its team is recognisable
  - per-species HP at turn 30 for common species (-1 when that species is not on the team)
"""
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# Constants and basic features for the building of the next ones
STATS = ["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]
BAD_STATUS = ["slp", "par", "tox", "psn", "frz", "brn"]
EFFECTS = ["substitute", "leechseed", "confusion", "trapped", "perish"]
KEY_MOVES = ["spikes", "rapidspin", "explosion", "selfdestruct", "curse", "bellydrum", "swordsdance",
             "growth", "rest", "sleeptalk", "sleeppowder", "lovelykiss", "hypnosis", "spore",
             "thunderwave", "toxic", "roar", "whirlwind"]
BOOST_STATS = ["atk", "def", "spa", "spd", "spe"]
CHECKPOINTS = [10, 20, 30] # Turns at which we change batch for comparing
LOW_HP = 0.25

# Moveset learning: keep moves a species uses in at least this share of its battles.
MOVESET_MIN_FRAC = 0.1

# p2 team completion: only trust a known full team that was seen this often and makes up
# this share of all known teams containing the revealed Pokemon (~94% precision when 5 are revealed).
TEAM_MIN_COUNT = 3
TEAM_MIN_SHARE = 0.8
MAX_KO_TURNS = 10.0


# Species seen in at least this many battles get their own "HP at turn 30" feature per side.
SPECIES_MIN_BATTLES = 300

# Classes of utility moves that will be counted per side
UTILITY_MOVES = {
    "explosion": {"explosion", "selfdestruct"},
    "rest": {"rest"},
    "sleeper": {"sleeppowder", "lovelykiss", "hypnosis", "spore", "sing"},
    "spiker": {"spikes"},
    "spinner": {"rapidspin"},
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

# Loading of the type advantages chart given in input for later accounting of type matchups
def load_type_chart(path):
    chart = defaultdict(lambda: 1.0)
    for row in pd.read_csv(path).itertuples(index=False):
        chart[(row.attacking_type, row.defending_type)] = float(row.multiplier)
    return chart

# Helper to compute the multiplier given by the boost stage
def boost_mult(stage):
    return (2 + stage) / 2 if stage >= 0 else 2 / (2 - stage)

# Helper used on both train and test data to have in a single place all the stats and types of the pokemon we could see
def build_pokedex(battles):
    dex = {}
    for b in battles:
        for p in b["p1_team_details"] + [b["p2_lead_details"]]:
            dex[p["name"]] = {"types": [t for t in p["types"] if t != "notype"], **{k: p[k] for k in STATS}}
    # Fallback for Pokemon only ever seen in p2 timelines: average stats, no types.
    mean = {k: float(np.mean([v[k] for v in dex.values()])) for k in STATS}
    dex["__unknown__"] = {"types": [], **mean}
    return dex

# Gets us the names of all the pokemon used by p2 during a battle
def revealed_p2(b):
    return list(dict.fromkeys(t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]))

# Gets how ofter a team of 6 is used by p2
def build_p2_team_index(battles):
    return Counter(frozenset(r) for r in map(revealed_p2, battles) if len(r) == 6)

# Tries to get what kind of moves a pokemon usually has
def build_movesets(battles):
    counts, appearances, details = defaultdict(Counter), Counter(), {}
    for b in battles:
        used = defaultdict(set)
        for t in b["battle_timeline"]:
            for side in ("p1", "p2"):
                m = t[f"{side}_move_details"]
                if m:
                    used[t[f"{side}_pokemon_state"]["name"]].add(m["name"])
                    details[m["name"]] = m
        for name, moves in used.items():
            appearances[name] += 1
            counts[name].update(moves)
    return {name: [details[m] for m, c in counts[name].items() if c >= MOVESET_MIN_FRAC * appearances[name]]
            for name in counts}

# Identifies the common species of pokemon that usually appear in the battles to find who are the most relevant ones
def build_common_species(battles):
    counts = Counter(n for b in battles for n in {p["name"] for p in b["p1_team_details"]} | set(revealed_p2(b)))
    return sorted(n for n, c in counts.items() if c >= SPECIES_MIN_BATTLES)

# Tries to get assuming a best scenario of perfect DVs and max stat experience
def stat_value(base, level, hp=False):
    core = (2 * base + 30 + 64) * level // 100
    return core + level + 10 if hp else core + 5

# Initialization of the class that will be used to create the features
class FeatureBuilder:
    def __init__(self, dex, chart, p2_teams=None, movesets=None, species=None):
        self.dex = dex
        self.chart = chart
        self.p2_teams = p2_teams or Counter()
        self.movesets = movesets or {}
        self.species = species or []
        self._best_hit = {}
        self._completed = {}

    # Gets teh informations about a pokemon giving its name
    def info(self, name):
        return self.dex.get(name, self.dex["__unknown__"])

    # Gets how effective a move is giving the typing of the move and the defender's types
    def eff(self, move_type, defender_types):
        m = 1.0
        for t in defender_types:
            m *= self.chart[(move_type, t)]
        return m

    # Calculates what is the best STAB multiplier agains given types of a defender
    def best_stab_eff(self, att_types, def_types):
        if not att_types:
            return 1.0
        return max(self.eff(t, def_types) for t in att_types)

    # Evaluates across all pairs of matchup pokemons between p1 and p2 who has teh advantage in the fight
    # log2 is used to have a linear scaling and then the mean is taken
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

    # Evaluates if a given pokemon is a special or physical attacker looking at its base stats
    def offensive_stat(self, name):
        i = self.info(name)
        return "atk" if i["base_atk"] >= i["base_spa"] else "spa"

    # Evaluates how much damage a move does giving in input the attacker, the defender, the move and all the possible boosts including STAB
    def move_damage(self, move, attacker, defender, att_boosts, def_boosts):
        if not move or move["base_power"] == 0 or move["category"] == "status":
            return 0.0, 1.0
        a, d = self.info(attacker), self.info(defender)
        if move["category"] == "physical":
            ratio = (a["base_atk"] * boost_mult(att_boosts["atk"])) / (d["base_def"] * boost_mult(def_boosts["def"]))
        else:
            ratio = (a["base_spa"] * boost_mult(att_boosts["spa"])) / (d["base_spd"] * boost_mult(def_boosts["spd"]))
        stab = 1.5 if move["type"] in a["types"] else 1.0
        e = self.eff(move["type"], d["types"])
        return move["base_power"] * stab * e * ratio * move["accuracy"], e

    # Tries to infer p2's full team. If it is already known, it returns it, otherwise it checks if it can be relibly inferred from
    # the common teams that are known
    def complete_p2(self, revealed):
        """p2's full team if the revealed Pokemon match one known team confidently, else None."""
        r = frozenset(revealed)
        if len(r) == 6:
            return r
        if r not in self._completed:
            matches = {team: n for team, n in self.p2_teams.items() if r <= team}
            top = max(matches, key=matches.get) if matches else None
            ok = top is not None and matches[top] >= TEAM_MIN_COUNT and \
                matches[top] / sum(matches.values()) >= TEAM_MIN_SHARE
            self._completed[r] = top if ok else None
        return self._completed[r]

    # Evaluates the expected damage as a fraction of the defender's max hp
    def hit_pct(self, move, attacker, defender, att_level, def_level):
        a, d = self.info(attacker), self.info(defender)
        atk, dfn = ("base_atk", "base_def") if move["category"] == "physical" else ("base_spa", "base_spd")
        dmg = (2 * att_level / 5 + 2) * move["base_power"] * stat_value(a[atk], att_level)
        dmg = dmg / stat_value(d[dfn], def_level) / 50 + 2
        dmg *= (1.5 if move["type"] in a["types"] else 1.0) * self.eff(move["type"], d["types"])
        return dmg * 0.92 * move["accuracy"] / stat_value(d["base_hp"], def_level, hp=True)

    # Evaluates what the best possible hit from an attacker would be from the given moveset, if not known, a generic 80 pwr STAB is assumed
    def best_hit(self, attacker, defender, att_level, def_level):
        key = (attacker, defender, att_level, def_level)
        if key not in self._best_hit:
            moves = [m for m in self.movesets.get(attacker, [])
                     if m["base_power"] > 0 and m["category"] != "status"]
            if not moves:
                types = self.info(attacker)["types"] or ["normal"]
                cat = "physical" if self.offensive_stat(attacker) == "atk" else "special"
                moves = [{"base_power": 80, "category": cat, "type": t, "accuracy": 1.0} for t in types]
            self._best_hit[key] = max(self.hit_pct(m, attacker, defender, att_level, def_level) for m in moves)
        return self._best_hit[key]

    # Evaluates how many hits an attacker would need to KO each defender, assuming the attacker always does its best hit
    def threat_features(self, f, side, attackers, defenders, def_hp, att_levels, def_levels):
        if not defenders:
            turns = [0.0]
        elif not attackers:
            turns = [MAX_KO_TURNS] * len(defenders)
        else:
            turns = [min(MAX_KO_TURNS, def_hp.get(d, 1.0) / max(
                max(self.best_hit(a, d, att_levels[a], def_levels[d]) for a in attackers), 1e-3))
                for d in defenders]
        f[f"{side}_ko_turns_total"] = sum(turns)
        f[f"{side}_ko_turns_mean"] = float(np.mean(turns))
        f[f"{side}_ko_turns_max"] = max(turns)
        f[f"{side}_n_one_hit_ko"] = sum(t <= 1 for t in turns) if defenders else 0

    # Counts, giving all the alive pokemons, how many utility moves are left and of which type
    def utility_counts(self, f, side, alive):
        for k, moves in UTILITY_MOVES.items():
            f[f"{side}_alive_{k}"] = sum(any(m["name"] in moves for m in self.movesets.get(n, [])) for n in alive)

    def team_keys(self, b):
        return frozenset(p["name"] for p in b["p1_team_details"]), self.complete_p2(revealed_p2(b))

    # Function that will build all the features, given a single battle and return them all in a dictionary
    def build(self, b):

        f = {}
        team1 = b["p1_team_details"]
        tl = b["battle_timeline"]
        names1 = [p["name"] for p in team1]

        # ---- static team features -------------------------------------------------

        p1_stats = np.array([[p[k] for k in STATS] for p in team1], dtype=float)
        p2_names = revealed_p2(b)
        completed = self.complete_p2(p2_names)
        p2_team = p2_names + sorted(set(completed) - set(p2_names)) if completed else p2_names
        p2_stats = np.array([[self.info(n)[k] for k in STATS] for n in p2_team], dtype=float)
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
        f["p2_team_completed"] = int(completed is not None and len(p2_names) < 6)
        f["type_matchup_all"] = self.matchup(names1, p2_team)
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
        spikes_on = {"p1": False, "p2": False}
        rested = {"p1": set(), "p2": set()}
        max_boost = {"p1": 0, "p2": 0}
        max_off_boost = {"p1": 0, "p2": 0}

        for t in tl:

            turn = t["turn"]
            batch = min((turn - 1) * len(CHECKPOINTS) // 30, len(CHECKPOINTS) - 1)
            s1, s2 = t["p1_pokemon_state"], t["p2_pokemon_state"]
            m1, m2 = t["p1_move_details"], t["p2_move_details"]
            n1, n2 = s1["name"], s2["name"]

            dmg_taken[batch] += max(hp1.get(n1, 1.0) - s1["hp_pct"], 0.0)
            dmg_dealt[batch] += max(hp2.get(n2, 1.0) - s2["hp_pct"], 0.0)
            hp1[n1], hp2[n2] = s1["hp_pct"], s2["hp_pct"]
            status1[n1], status2[n2] = s1["status"], s2["status"]
            if s1["status"] == "fnt" and n1 not in fainted1:
                fainted1.add(n1); ko_by_batch1[batch] += 1; last_ko1 = turn
            if s2["status"] == "fnt" and n2 not in fainted2:
                fainted2.add(n2); ko_by_batch2[batch] += 1; last_ko2 = turn

            # Null move: a switch if the active Pokemon changed, otherwise it flinched or something similar.
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
                max_off_boost[side] = max(max_off_boost[side], s["boosts"][self.offensive_stat(n)])
            mv["p1_spikes_turns"] += spikes_on["p1"]
            mv["p2_spikes_turns"] += spikes_on["p2"]
            d1, e1 = self.move_damage(m1, n1, n2, s1["boosts"], s2["boosts"])
            d2, e2 = self.move_damage(m2, n2, n1, s2["boosts"], s1["boosts"])
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
                # Unseen p2 Pokemon are obviously assumed to be at full hp
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
        # alive2 includes inferred unseen pokemons while hp2 only has the revealed ones.
        alive2 = [n for n in p2_team if n not in fainted2]
        f["p1_alive"] = len(alive1)
        f["p2_alive_seen"] = sum(n not in fainted2 for n in p2_names)
        f["p2_alive_est"] = 6 - len(fainted2)
        f["p1_low_hp"] = sum(0 < hp1[n] < LOW_HP for n in alive1)
        f["p2_low_hp"] = sum(0 < hp2.get(n, 1.0) < LOW_HP for n in alive2)
        f["p1_alive_hp_mean"] = np.mean([hp1[n] for n in alive1]) if alive1 else 0.0
        f["p2_alive_hp_mean"] = np.mean([hp2.get(n, 1.0) for n in alive2]) if alive2 else 0.0
        f["type_matchup_alive"] = self.matchup(alive1, alive2)

        levels1 = {p["name"]: p["level"] for p in team1}
        levels2 = defaultdict(lambda: 100, {b["p2_lead_details"]["name"]: b["p2_lead_details"]["level"]})
        self.threat_features(f, "p1", alive1, alive2, hp2, levels1, levels2)
        self.threat_features(f, "p2", alive2, alive1, hp1, levels2, levels1)
        f["ko_turns_total_diff"] = f["p2_ko_turns_total"] - f["p1_ko_turns_total"]
        f["ko_turns_mean_diff"] = f["p2_ko_turns_mean"] - f["p1_ko_turns_mean"]
        f["one_hit_ko_diff"] = f["p1_n_one_hit_ko"] - f["p2_n_one_hit_ko"]
        self.utility_counts(f, "p1", alive1)
        self.utility_counts(f, "p2", alive2)

        # HP of each common species at turn 30 (0 = fainted, -1 = not on that team / not known).
        team2 = set(p2_team)
        for n in self.species:
            f[f"sp1_{n}"] = hp1.get(n, -1.0)
            f[f"sp2_{n}"] = hp2.get(n, 1.0) if n in team2 else -1.0
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
        # Per-stat boosts, plus the boost on the stat each active Pokemon actually attacks with and the average of the two defenses boosts
        for side, a in (("p1", a1), ("p2", a2)):
            for st in BOOST_STATS:
                f[f"{side}_boost_{st}"] = a["boosts"][st]
            f[f"{side}_off_boost"] = a["boosts"][self.offensive_stat(a["name"])]
            f[f"{side}_def_boost"] = (a["boosts"]["def"] + a["boosts"]["spd"]) / 2
        f["off_boost_diff"] = f["p1_off_boost"] - f["p2_off_boost"]
        f["def_boost_diff"] = f["p1_def_boost"] - f["p2_def_boost"]
        f["spe_boost_diff"] = f["p1_boost_spe"] - f["p2_boost_spe"]
        f["active_eff_speed_diff"] = (self.info(a1["name"])["base_spe"] * boost_mult(a1["boosts"]["spe"])
                                      - self.info(a2["name"])["base_spe"] * boost_mult(a2["boosts"]["spe"]))
        for e in EFFECTS:
            f[f"p1_eff_{e}"] = int(e in a1["effects"])
            f[f"p2_eff_{e}"] = int(e in a2["effects"])

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
            f[f"{side}_max_off_boost"] = max_off_boost[side]
        f["spikes_diff"] = f["p2_spikes_on_side"] - f["p1_spikes_on_side"]
        f["p1_rested_alive"] = len(rested["p1"] & set(alive1))
        f["p2_rested_alive"] = len(rested["p2"] & set(alive2))
        f["est_dmg_diff"] = f["p1_est_dmg"] - f["p2_est_dmg"]
        f["mean_eff_diff"] = f["p1_mean_eff"] - f["p2_mean_eff"]
        f["cant_move_diff"] = f["p2_cant_move"] - f["p1_cant_move"]
        return f


def make_features(train, test, type_chart_path, return_keys=False):
    battles = train + test
    fb = FeatureBuilder(build_pokedex(battles), load_type_chart(type_chart_path),
                        build_p2_team_index(battles), build_movesets(battles), build_common_species(battles))
    X_train = pd.DataFrame([fb.build(b) for b in train])
    X_test = pd.DataFrame([fb.build(b) for b in test])
    if return_keys:
        return X_train, X_test, [fb.team_keys(b) for b in train], [fb.team_keys(b) for b in test]
    return X_train, X_test
