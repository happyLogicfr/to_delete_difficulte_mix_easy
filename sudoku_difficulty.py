# sudoku_difficulty.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional, Callable
import random
from copy import deepcopy
import hashlib

from sudoku_hash_db import load_global_hashes, save_global_hashes

Grid = List[List[int]]


# ---------- Utils de hash / représentation ----------

def canon_str(grid: Grid) -> str:
    """Chaîne canonique pour une grille (ligne par ligne)."""
    return "".join("".join(str(v) for v in row) for row in grid)


def hash_grid_sha256(grid: Grid) -> str:
    """Hash hex (64) d'une grille basée sur canon_str (exact match)."""
    return hashlib.sha256(canon_str(grid).encode("utf-8")).hexdigest().lower()


def book_hash_v1(puzzles: List[Tuple[Grid, Grid]]) -> str:
    """
    Hash d'ensemble indépendant de l'ordre :
    - hash de chaque puzzle,
    - tri,
    - payload versionné,
    - re-hash.
    """
    per = sorted(hash_grid_sha256(p) for (p, _s) in puzzles)
    payload = "sudoku-book:v1\ncount=" + str(len(per)) + "\n" + "\n".join(per) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().lower()


# ====================================================
#   SOLVEUR / UNICITÉ — BITSETS (rapide)
# ====================================================

FULL_MASK = (1 << 9) - 1  # 9 bits


def _box_idx(r: int, c: int) -> int:
    return (r // 3) * 3 + (c // 3)


def _init_masks_bitset(grid: Grid):
    row_used = [0] * 9
    col_used = [0] * 9
    box_used = [0] * 9
    empties = []
    for r in range(9):
        for c in range(9):
            v = grid[r][c]
            if v:
                b = 1 << (v - 1)
                row_used[r] |= b
                col_used[c] |= b
                box_used[_box_idx(r, c)] |= b
            else:
                empties.append((r, c))
    return row_used, col_used, box_used, empties


def _allowed_mask(row_used, col_used, box_used, r: int, c: int) -> int:
    return FULL_MASK ^ (row_used[r] | col_used[c] | box_used[_box_idx(r, c)])


def _popcount(x: int) -> int:
    try:
        return x.bit_count()
    except AttributeError:  # Python <3.8
        return bin(x).count("1")


def _count_solutions_bitset(grid: Grid, limit: int = 2) -> int:
    row_used, col_used, box_used, empties = _init_masks_bitset(grid)

    def mrv_key(rc):
        r, c = rc
        return (_popcount(_allowed_mask(row_used, col_used, box_used, r, c)), r, c)

    empties.sort(key=mrv_key)
    sols = 0

    def dfs(k: int = 0):
        nonlocal sols
        if sols >= limit:
            return
        if k == len(empties):
            sols += 1
            return
        r, c = empties[k]
        cand = _allowed_mask(row_used, col_used, box_used, r, c)
        if cand == 0:
            return
        bidx = _box_idx(r, c)
        x = cand
        while x:
            lsb = x & -x
            x ^= lsb
            row_used[r] |= lsb
            col_used[c] |= lsb
            box_used[bidx] |= lsb
            dfs(k + 1)
            row_used[r] ^= lsb
            col_used[c] ^= lsb
            box_used[bidx] ^= lsb
            if sols >= limit:
                return

    dfs(0)
    return sols


def has_unique_solution(grid: Grid) -> bool:
    return _count_solutions_bitset(deepcopy(grid), limit=2) == 1


def count_solutions(grid: Grid, limit: int = 2) -> int:
    return _count_solutions_bitset(deepcopy(grid), limit=limit)


# ====================================================
#   GÉNÉRATION D'UNE GRILLE COMPLÈTE
# ====================================================

def generate_full_grid() -> Grid:
    grid: Grid = [[0] * 9 for _ in range(9)]

    def backtrack(r=0, c=0) -> bool:
        if r == 9:
            return True
        nr, nc = (r, c + 1) if c < 8 else (r + 1, 0)
        if grid[r][c] != 0:
            return backtrack(nr, nc)
        vals = list(range(1, 10))
        random.shuffle(vals)
        for v in vals:
            if all(grid[r][x] != v for x in range(9)) and all(grid[x][c] != v for x in range(9)):
                br, bc = 3 * (r // 3), 3 * (c // 3)
                if all(grid[rr][cc] != v for rr in range(br, br + 3) for cc in range(bc, bc + 3)):
                    grid[r][c] = v
                    if backtrack(nr, nc):
                        return True
                    grid[r][c] = 0
        return False

    backtrack()
    return grid


# ====================================================
#   CANDIDATS & UNITÉS
# ====================================================

UNITS = []
for r in range(9):
    UNITS.append([(r, c) for c in range(9)])
for c in range(9):
    UNITS.append([(r, c) for r in range(9)])
for br in range(0, 9, 3):
    for bc in range(0, 9, 3):
        UNITS.append([(br + dr, bc + dc) for dr in range(3) for dc in range(3)])


def grid_candidates(grid: Grid) -> Dict[Tuple[int, int], Set[int]]:
    cands: Dict[Tuple[int, int], Set[int]] = {}
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                used = set(grid[r]) | {grid[i][c] for i in range(9)}
                br, bc = 3 * (r // 3), 3 * (c // 3)
                used |= {grid[i][j] for i in range(br, br + 3) for j in range(bc, bc + 3)}
                cands[(r, c)] = {v for v in range(1, 10) if v not in used}
    return cands


# ====================================================
#   STRATÉGIES LOGIQUES — COMMUNES
# ====================================================

def apply_singletons(grid: Grid, cands: Dict[Tuple[int, int], Set[int]]):
    """
    Naked singles + hidden singles. Retourne (placements, None).
    """
    changes = 0

    # Naked singles
    to_set = []
    for rc, opts in cands.items():
        if len(opts) == 1:
            (v,) = tuple(opts)
            to_set.append((rc, v))
    for (r, c), v in to_set:
        grid[r][c] = v
        changes += 1

    if changes:
        return changes, None  # on recalculera les candidats hors de cette fn

    # Hidden singles
    for unit in UNITS:
        positions_by_val = {v: [] for v in range(1, 10)}
        for (r, c) in unit:
            if grid[r][c] == 0:
                for v in cands[(r, c)]:
                    positions_by_val[v].append((r, c))
        for v, places in positions_by_val.items():
            if len(places) == 1:
                (r, c) = places[0]
                grid[r][c] = v
                return 1, None
    return 0, None


def apply_naked_pairs(grid: Grid, cands: Dict[Tuple[int, int], Set[int]]):
    """
    Naked pairs dans une unité : élimination ailleurs dans l’unité.
    Retourne (nb_elims, events)
    """
    removed = 0
    events = []
    for unit in UNITS:
        pairs: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for (r, c) in unit:
            if grid[r][c] == 0 and len(cands[(r, c)]) == 2:
                key = tuple(sorted(cands[(r, c)]))
                pairs.setdefault(key, []).append((r, c))
        for key, cells in pairs.items():
            if len(cells) == 2:
                for (r, c) in unit:
                    if grid[r][c] == 0 and (r, c) not in cells:
                        before = len(cands[(r, c)])
                        newset = cands[(r, c)] - set(key)
                        if len(newset) < before:
                            cands[(r, c)] = newset
                            removed += (before - len(newset))
                            events.append(("naked_pair", key, [tuple(x) for x in cells], (r + 1, c + 1)))
    return removed, events


def apply_hidden_pairs(grid: Grid, cands: Dict[Tuple[int, int], Set[int]]):
    """
    Hidden pairs. Retourne (nb_elims, events)
    """
    removed = 0
    events = []
    for unit in UNITS:
        pos_by_val = {v: [] for v in range(1, 10)}
        empties = [(r, c) for (r, c) in unit if grid[r][c] == 0]
        for (r, c) in empties:
            for v in cands[(r, c)]:
                pos_by_val[v].append((r, c))
        vals = [v for v in range(1, 10) if 1 <= len(pos_by_val[v]) <= 2]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                v1, v2 = vals[i], vals[j]
                if len(pos_by_val[v1]) == 2 and pos_by_val[v1] == pos_by_val[v2]:
                    cells = pos_by_val[v1]
                    for (r, c) in cells:
                        before = set(cands[(r, c)])
                        keep = {v1, v2}
                        if before - keep:
                            cands[(r, c)] = keep
                            removed += len(before - keep)
                            events.append(("hidden_pair", (v1, v2), (r + 1, c + 1)))
    return removed, events


def apply_pointing_box_line(grid: Grid, cands: Dict[Tuple[int, int], Set[int]]):
    """
    Pointing pairs/triples (box-line reduction).
    Retourne (nb_elims, events)
    """
    removed = 0
    events = []
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = [(br + dr, bc + dc) for dr in range(3) for dc in range(3)]
            for v in range(1, 10):
                pos = [(r, c) for (r, c) in box if grid[r][c] == 0 and v in cands[(r, c)]]
                if not pos:
                    continue
                rows = {r for (r, _c) in pos}
                cols = {c for (_r, c) in pos}
                if len(rows) == 1:
                    r = next(iter(rows))
                    for c in range(9):
                        if not (br <= r < br + 3 and bc <= c < bc + 3) and grid[r][c] == 0 and v in cands[(r, c)]:
                            cands[(r, c)].discard(v)
                            removed += 1
                            events.append(("locked_row_claiming", v, (br // 3 + 1, bc // 3 + 1), (r + 1, c + 1)))
                if len(cols) == 1:
                    c = next(iter(cols))
                    for r in range(9):
                        if not (br <= r < br + 3 and bc <= c < bc + 3) and grid[r][c] == 0 and v in cands[(r, c)]:
                            cands[(r, c)].discard(v)
                            removed += 1
                            events.append(("locked_col_claiming", v, (br // 3 + 1, bc // 3 + 1), (r + 1, c + 1)))
    return removed, events


def xwing_eliminations(grid: Grid, cands: Dict[Tuple[int, int], Set[int]]):
    """
    X-Wing sur lignes et colonnes.
    Retourne (nb_elims, events)
    """
    removed = 0
    events = []
    # lignes
    for v in range(1, 10):
        row_cols = []
        for r in range(9):
            cols = [c for c in range(9) if grid[r][c] == 0 and v in cands[(r, c)]]
            if len(cols) == 2:
                row_cols.append((r, tuple(cols)))
        for i in range(len(row_cols)):
            for j in range(i + 1, len(row_cols)):
                r1, cols1 = row_cols[i]
                r2, cols2 = row_cols[j]
                if cols1 == cols2:
                    c1, c2 = cols1
                    for rr in range(9):
                        if rr != r1 and rr != r2:
                            for cc in (c1, c2):
                                if grid[rr][cc] == 0 and v in cands[(rr, cc)]:
                                    cands[(rr, cc)].discard(v)
                                    removed += 1
                                    events.append(("xwing_row", v, (r1 + 1, r2 + 1), (c1 + 1, c2 + 1), (rr + 1, cc + 1)))
    # colonnes
    for v in range(1, 10):
        col_rows = []
        for c in range(9):
            rows = [r for r in range(9) if grid[r][c] == 0 and v in cands[(r, c)]]
            if len(rows) == 2:
                col_rows.append((c, tuple(rows)))
        for i in range(len(col_rows)):
            for j in range(i + 1, len(col_rows)):
                c1, rows1 = col_rows[i]
                c2, rows2 = col_rows[j]
                if rows1 == rows2:
                    r1, r2 = rows1
                    for cc in range(9):
                        if cc != c1 and cc != c2:
                            for rr in (r1, r2):
                                if grid[rr][cc] == 0 and v in cands[(rr, cc)]:
                                    cands[(rr, cc)].discard(v)
                                    removed += 1
                                    events.append(("xwing_col", v, (r1 + 1, r2 + 1), (c1 + 1, c2 + 1), (rr + 1, cc + 1)))
    return removed, events


# ====================================================
#   FACILE — singles only + bonne répartition
# ====================================================

def _human_solve_basic_only(puzzle: Grid) -> Tuple[bool, bool]:
    """
    Resolveur très simple : naked + hidden singles uniquement.
    Retourne (solved, chain_ok).
    """
    g = deepcopy(puzzle)
    chain_ok = True

    while True:
        cands = grid_candidates(g)
        if not cands:
            # plus de zéros
            return True, chain_ok

        # singles only
        naked_moves = {pos: next(iter(vals)) for pos, vals in cands.items() if len(vals) == 1}

        # hidden singles
        hidden_moves = {}
        for unit in UNITS:
            freq = {d: [] for d in range(1, 10)}
            for (r, c) in unit:
                if g[r][c] == 0:
                    for d in cands[(r, c)]:
                        freq[d].append((r, c))
            for d, cells in freq.items():
                if len(cells) == 1:
                    hidden_moves[cells[0]] = d

        moves = {**hidden_moves, **naked_moves}
        if not moves:
            return False, False

        for (r, c), v in moves.items():
            g[r][c] = v

        # vérification de contradictions
        for (r, c), vals in grid_candidates(g).items():
            if len(vals) == 0 and g[r][c] == 0:
                return False, False


def _well_distributed(puzzle: Grid, min_row: int = 4, min_col: int = 4, min_block: int = 3) -> bool:
    for r in range(9):
        if sum(1 for v in puzzle[r] if v != 0) < min_row:
            return False
    for c in range(9):
        if sum(1 for r in range(9) if puzzle[r][c] != 0) < min_col:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            cnt = 0
            for dr in range(3):
                for dc in range(3):
                    if puzzle[br + dr][bc + dc] != 0:
                        cnt += 1
            if cnt < min_block:
                return False
    return True


def _generate_puzzle_easy_variant(
    *,
    clues_min: int,
    clues_max: int,
    min_first_sweep: Optional[int] = None,
    max_first_sweep: Optional[int] = None,
    max_avg_candidates: Optional[float] = None,
) -> Tuple[Grid, Grid]:
    """
    Génère un puzzle FACILE (singles only) avec contraintes paramétrables.

    Contraintes communes (facile) :
      - solution unique
      - résoluble uniquement avec singles (naked + hidden)
      - bonne répartition (lignes/colonnes/blocs)

    Contraintes optionnelles pour sous-niveaux :
      - min_first_sweep / max_first_sweep : nb de placements obtenus au 1er balayage singles
      - max_avg_candidates : moyenne de candidats au départ (plus bas => plus facile)
    """
    while True:
        full = generate_full_grid()
        puzzle = deepcopy(full)

        target = random.randint(clues_min, clues_max)
        cells = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(cells)

        def n_clues(g: Grid) -> int:
            return sum(1 for r in range(9) for c in range(9) if g[r][c] != 0)

        tries = 0
        max_tries = 5000
        while n_clues(puzzle) > target and tries < max_tries:
            tries += 1
            r, c = random.choice(cells)
            if puzzle[r][c] == 0:
                continue
            keep = puzzle[r][c]
            puzzle[r][c] = 0

            if not has_unique_solution(puzzle):
                puzzle[r][c] = keep
                continue

            solved, chain_ok = _human_solve_basic_only(puzzle)
            if not (solved and chain_ok):
                puzzle[r][c] = keep
                continue

            if not _well_distributed(puzzle):
                puzzle[r][c] = keep
                continue

        clues = n_clues(puzzle)
        if not (clues_min <= clues <= clues_max):
            continue

        # Sous-filtres (optionnels)
        if max_avg_candidates is not None:
            avg0 = average_candidates(grid_candidates(puzzle))
            if avg0 > max_avg_candidates:
                continue

        if min_first_sweep is not None or max_first_sweep is not None:
            fs = first_sweep_singles_count(puzzle)
            if min_first_sweep is not None and fs < min_first_sweep:
                continue
            if max_first_sweep is not None and fs > max_first_sweep:
                continue

        return puzzle, full


def _generate_puzzle_easy_plus() -> Tuple[Grid, Grid]:
    """Très facile / débutant."""
    return _generate_puzzle_easy_variant(
        clues_min=52,
        clues_max=58,
        min_first_sweep=25,
        max_avg_candidates=2.2,
    )


def _generate_puzzle_easy(
    clues_min: int = 45,
    clues_max: int = 55,
    max_restarts: int = 200,
    max_tries_per_restart: int = 6000,
) -> Tuple[Grid, Grid]:
    """
    Génère un puzzle FACILE :
      - indices nombreux (clues_min..clues_max),
      - résoluble uniquement avec singles (naked + hidden),
      - bien réparti.

    IMPORTANT : version bornée (pas de récursion infinie).
    """
    for _restart in range(max_restarts):
        full = generate_full_grid()
        puzzle = deepcopy(full)

        target = random.randint(clues_min, clues_max)
        cells = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(cells)

        def n_clues(g: Grid) -> int:
            return sum(1 for r in range(9) for c in range(9) if g[r][c] != 0)

        tries = 0
        while n_clues(puzzle) > target and tries < max_tries_per_restart:
            tries += 1
            r, c = random.choice(cells)
            if puzzle[r][c] == 0:
                continue

            keep = puzzle[r][c]
            puzzle[r][c] = 0

            # unicité
            if not has_unique_solution(puzzle):
                puzzle[r][c] = keep
                continue

            # solvable singles-only
            solved, chain_ok = _human_solve_basic_only(puzzle)
            if not (solved and chain_ok):
                puzzle[r][c] = keep
                continue

            # bonne répartition
            if not _well_distributed(puzzle):
                puzzle[r][c] = keep
                continue

        clues = n_clues(puzzle)
        if clues_min <= clues <= clues_max:
            return puzzle, full

    raise RuntimeError(
        f"Impossible de générer un puzzle easy (clues {clues_min}..{clues_max}) "
        f"après {max_restarts} redémarrages."
    )


def _generate_puzzle_easy_plus() -> Tuple[Grid, Grid]:
    """Facile+ (débutant) : plus d'indices, plus fluide."""
    return _generate_puzzle_easy(clues_min=52, clues_max=58)


def _generate_puzzle_easy_standard() -> Tuple[Grid, Grid]:
    """Facile (standard) : profil historique."""
    return _generate_puzzle_easy(clues_min=45, clues_max=55)


def _generate_puzzle_easy_minus() -> Tuple[Grid, Grid]:
    """
    Facile− (transition) : toujours singles-only, mais fenêtre d'indices
    volontairement un peu plus haute que 42–47 pour éviter les cas rares
    qui bloquent la génération.
    """
    return _generate_puzzle_easy(clues_min=45, clues_max=50)



# ====================================================
#   MOYEN — stratégies intermédiaires
# ====================================================
# — stratégies intermédiaires
# ====================================================

def count_clues(grid: Grid) -> int:
    return sum(1 for row in grid for v in row if v)


def block_counts(grid: Grid) -> List[int]:
    counts = []
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            counts.append(sum(1 for dr in range(3) for dc in range(3) if grid[br + dr][bc + dc]))
    return counts


def min_line_counts(grid: Grid) -> Tuple[int, int]:
    rows = [sum(1 for v in row if v) for row in grid]
    cols = [sum(1 for r in range(9) if grid[r][c]) for c in range(9)]
    return min(rows), min(cols)


def any_line_col_block_le2(grid: Grid) -> bool:
    row_mins, col_mins = min_line_counts(grid)
    if row_mins <= 2 or col_mins <= 2:
        return True
    return any(cnt <= 2 for cnt in block_counts(grid))


def band_stack_bottleneck(grid: Grid) -> bool:
    """
    Détecte un band/stack où un bloc ≤2 et les deux autres bloc sont aussi "pauvres".
    Ici, on marque comme goulot si un bloc ≤2 ET la somme des 2 autres ≤6.
    """
    # bands (3 lignes x 3 colonnes en blocs)
    blocks = [
        [sum(1 for dr in range(3) for dc in range(3) if grid[br + dr][bc + dc]) for bc in range(0, 9, 3)]
        for br in range(0, 9, 3)
    ]
    # test bands
    for band in blocks:
        for i in range(3):
            if band[i] <= 2 and (band[(i + 1) % 3] + band[(i + 2) % 3]) <= 6:
                return True
    # stacks: transpose l'indice
    stacks = [
        [sum(1 for dr in range(3) for dc in range(3) if grid[br + dr][bc + dc]) for br in range(0, 9, 3)]
        for bc in range(0, 9, 3)
    ]
    for stack in stacks:
        for i in range(3):
            if stack[i] <= 2 and (stack[(i + 1) % 3] + stack[(i + 2) % 3]) <= 6:
                return True
    return False


def average_candidates(cands: Dict[Tuple[int, int], Set[int]]) -> float:
    if not cands:
        return 0.0
    total = sum(len(s) for s in cands.values())
    return float(total) / float(len(cands))


def first_sweep_singles_count(puzzle: Grid) -> int:
    """Nombre de placements obtenus uniquement via singles + hidden singles sur le départ."""
    g = deepcopy(puzzle)
    placed = 0
    while True:
        cands = grid_candidates(g)
        if not cands:
            break
        chg, _ = apply_singletons(g, cands)
        if chg == 0:
            break
        placed += chg
    return placed


def early_pairs_efficiency(puzzle: Grid) -> Tuple[int, int]:
    """
    Après le premier balayage, combien d'éliminations via pairs/pointing pour débloquer 1–2 cases.
    Retourne (elims, placements).
    """
    g = deepcopy(puzzle)
    # premier balayage
    while True:
        cands = grid_candidates(g)
        if not cands:
            return 0, 0
        chg, _ = apply_singletons(g, cands)
        if chg == 0:
            break
    # boucle "interactions" jusqu'à obtenir jusqu'à 2 placements ou blocage
    elims = 0
    placements = 0
    for _ in range(200):  # borne de sécurité
        cands = grid_candidates(g)
        if not cands:
            break
        e1, _ = apply_pointing_box_line(g, cands)
        e2, _ = apply_naked_pairs(g, cands)
        e3, _ = apply_hidden_pairs(g, cands)
        elims += (e1 + e2 + e3)
        # puis essayer de convertir en placements
        chg, _ = apply_singletons(g, cands)
        if chg:
            placements += chg
            if placements >= 2:
                break
        elif (e1 + e2 + e3) == 0:
            break
    return elims, placements


def solve_with_intermediate_strategies(puzzle: Grid):
    """
    Tente de résoudre uniquement avec : singles, naked/hidden pairs, pointing.
    Retourne (solved: bool, scores: dict, needed_advanced: bool).
    """
    g = deepcopy(puzzle)
    scores = {"single": 0, "naked_pair": 0, "hidden_pair": 0, "pointing": 0}
    needed_advanced = False

    while True:
        cands = grid_candidates(g)

        if not cands:  # plus de zéros -> résolu
            return True, scores, False

        progressed = False

        # 1) Singles
        chg, _ = apply_singletons(g, cands)
        if chg:
            scores["single"] += chg
            progressed = True
            continue  # recalcul complet au tour suivant

        # 2) Pointing box-line
        rem, _ = apply_pointing_box_line(g, cands)
        if rem:
            scores["pointing"] += rem
            progressed = True

        # 3) Naked pairs
        rem, _ = apply_naked_pairs(g, cands)
        if rem:
            scores["naked_pair"] += rem
            progressed = True

        # 4) Hidden pairs
        rem, _ = apply_hidden_pairs(g, cands)
        if rem:
            scores["hidden_pair"] += rem
            progressed = True

        if not progressed:
            needed_advanced = any(0 in row for row in g)
            return all(0 not in row for row in g), scores, needed_advanced


def difficulty_score(scores: dict) -> int:
    return (
        scores.get("single", 0) * 1
        + scores.get("pointing", 0) * 3
        + scores.get("naked_pair", 0) * 4
        + scores.get("hidden_pair", 0) * 5
    )


def classify_medium(puzzle: Grid) -> bool:
    """
    True si :
      - 38..44 indices
      - solvable SANS techniques avancées (seulement celles codées)
      - premier balayage (singles/hidden singles) ≥ 8 placements
      - moyenne des candidats après annotation ≤ 3.2
      - chaque bloc 3×3 a ≥ 3 indices ; aucun row/col/bloc ≤ 2
      - pas de goulot band/stack
      - score global dans une fourchette raisonnable
    """
    clues = count_clues(puzzle)
    if not (38 <= clues <= 44):
        return False

    # Blocs minimums
    bl = block_counts(puzzle)
    if any(cnt < 3 for cnt in bl):
        return False

    # Lignes/colonnes/blocs trop vides ?
    if any_line_col_block_le2(puzzle):
        return False

    # Goulots d'étranglement band/stack
    if band_stack_bottleneck(puzzle):
        return False

    # Candidats moyens au départ
    c0 = grid_candidates(puzzle)
    avg0 = average_candidates(c0)
    if avg0 > 3.2:
        return False

    # Premier balayage : singles only
    first_sw = first_sweep_singles_count(puzzle)
    if first_sw < 8:
        return False

    # Efficacité des paires au tout début : si beaucoup d'éliminations pour 0–1 placement, ça penche "difficile"
    elims, places = early_pairs_efficiency(puzzle)
    if places <= 1 and elims >= 12:
        return False

    # Résolubilité avec notre moteur intermédiaire
    solved, scores, needed_advanced = solve_with_intermediate_strategies(puzzle)
    if not solved or needed_advanced:
        return False

    score = difficulty_score(scores)
    if not (36 <= score <= 160):
        return False

    return True


def _generate_puzzle_medium() -> Tuple[Grid, Grid]:
    """
    Génère une grille intermédiaire (moyenne) en filtrant avec classify_medium.
    """
    max_tries = 10000
    tries = 0
    while True:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("Impossible de générer un puzzle medium dans les limites")

        full = generate_full_grid()
        puzzle = deepcopy(full)

        # cible 38..44 indices
        cells = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(cells)
        clues_target = random.randint(38, 44)
        clues = 81
        for (r, c) in cells:
            if clues <= clues_target:
                break
            keep = puzzle[r][c]
            puzzle[r][c] = 0
            if has_unique_solution(puzzle):
                clues -= 1
            else:
                puzzle[r][c] = keep

        if not classify_medium(puzzle):
            continue

        return puzzle, full


# ====================================================
#   DIFFICILE — solveur logique avancé (X-Wing inclus)
# ====================================================

def too_many_immediate_singles(puzzle: Grid, max_immediate: int = 10) -> bool:
    g = deepcopy(puzzle)
    placed = 0
    while True:
        cands = grid_candidates(g)
        if not cands:
            break
        chg, _ = apply_singletons(g, cands)
        if chg == 0:
            break
        placed += chg
        if placed > max_immediate:
            return True
    return False


def logical_step(grid: Grid, cands: Dict[Tuple[int, int], Set[int]]):
    stats = {"single": 0, "locked": 0, "naked_pair": 0, "hidden_pair": 0, "xwing": 0}
    events = {"locked": [], "hidden_pair": [], "xwing": []}
    chg, _ = apply_singletons(grid, cands)
    if chg:
        stats["single"] += chg
        return True, stats, events
    rem, ev = apply_pointing_box_line(grid, cands)
    if rem:
        stats["locked"] += rem
        events["locked"].extend(ev)
    rem, _ = apply_naked_pairs(grid, cands)
    if rem:
        stats["naked_pair"] += rem
    rem, ev = apply_hidden_pairs(grid, cands)
    if rem:
        stats["hidden_pair"] += rem
        events["hidden_pair"].extend(ev)
    rem, ev = xwing_eliminations(grid, cands)
    if rem:
        stats["xwing"] += rem
        events["xwing"].extend(ev)
    if any(stats[k] for k in ("locked", "naked_pair", "hidden_pair", "xwing")):
        chg, _ = apply_singletons(grid, cands)
        if chg:
            stats["single"] += chg
        return True, stats, events
    return False, stats, events


def solve_logically_no_guess(grid: Grid):
    g = deepcopy(grid)
    used = {"single": 0, "locked": 0, "naked_pair": 0, "hidden_pair": 0, "xwing": 0}
    evidence = {"locked": [], "hidden_pair": [], "xwing": []}
    for _ in range(600):
        cands = grid_candidates(g)
        if not cands:
            return True, used, evidence
        progressed, stats, ev = logical_step(g, cands)
        for k in used:
            used[k] += stats.get(k, 0)
        for k in evidence:
            evidence[k].extend(ev.get(k, []))
        if not progressed:
            return False, used, evidence
    return False, used, evidence


def classify_difficult_human(puzzle: Grid):
    """
    Variante 'difficile faisable' (human-friendly) :
      - solution unique
      - >= 32 indices (évite les grilles trop maigres)
      - <= 10 singles immédiats au départ (départ pas trop sec)
      - résolution 100% logique (solveur actuel)
      - singles globaux suffisants (>= 12) pour un bon rythme
      - un peu de locked/hidden pairs
      - X-Wing optionnel mais pas en rafale (<= 6 éliminations)
    """
    if count_clues(puzzle) < 32:
        return False, None
    if count_solutions(puzzle, limit=2) != 1:
        return False, None
    if too_many_immediate_singles(puzzle, max_immediate=10):
        return False, None
    solved, used, evidence = solve_logically_no_guess(puzzle)
    if not solved:
        return False, None
    if used.get("single", 0) < 12:
        return False, None
    if (used.get("locked", 0) + used.get("hidden_pair", 0)) <= 0:
        return False, None
    if used.get("xwing", 0) > 6:
        return False, None
    return True, {"used": used, "evidence": evidence}


def _generate_puzzle_hard() -> Tuple[Grid, Grid]:
    """
    Génère un puzzle difficile 'faisable' (profil hard).
    """
    max_tries = 5000
    tries = 0

    while True:
        tries += 1
        if tries > max_tries:
            raise RuntimeError("Impossible de générer un puzzle difficile dans les limites")

        full = generate_full_grid()
        puzzle = deepcopy(full)

        # suppression redistribuée (cible confortable 32..40 indices)
        cells = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(cells)
        clues_target = random.randint(32, 40)
        clues = 81
        for (r, c) in cells:
            if clues <= clues_target:
                break
            keep = puzzle[r][c]
            puzzle[r][c] = 0
            if count_solutions(puzzle, limit=2) == 1:
                clues -= 1
            else:
                puzzle[r][c] = keep

        ok, meta = classify_difficult_human(puzzle)
        if not ok:
            continue

        return puzzle, full


# ====================================================
#   PROFILS & GÉNÉRATION MULTI-PUZZLES
# ====================================================

@dataclass
class DifficultyProfile:
    name: str
    generator: Callable[[], Tuple[Grid, Grid]]


# profils concrets
EASY_PLUS_PROFILE = DifficultyProfile("easy_plus", _generate_puzzle_easy_plus)
EASY_PROFILE = DifficultyProfile("easy", _generate_puzzle_easy_standard)
EASY_MINUS_PROFILE = DifficultyProfile("easy_minus", _generate_puzzle_easy_minus)
MEDIUM_PROFILE = DifficultyProfile("medium", _generate_puzzle_medium)
HARD_PROFILE = DifficultyProfile("hard", _generate_puzzle_hard)

PROFILES: Dict[str, DifficultyProfile] = {
    EASY_PLUS_PROFILE.name: EASY_PLUS_PROFILE,
    EASY_PROFILE.name: EASY_PROFILE,
    EASY_MINUS_PROFILE.name: EASY_MINUS_PROFILE,
    MEDIUM_PROFILE.name: MEDIUM_PROFILE,
    HARD_PROFILE.name: HARD_PROFILE,
}



def generate_puzzles_for_profile(profile: DifficultyProfile, count: int) -> List[Tuple[Grid, Grid]]:
    """
    Génère `count` puzzles pour un profil donné, en garantissant :
      - pas de doublons dans la génération courante
      - pas de doublons vis-à-vis de l'historique global
        (fichier puzzle_hashes_all.txt, via sudoku_hash_db).
    """
    # Historique global (tous livres / toutes difficultés confondus)
    global_hashes = load_global_hashes()

    puzzles: List[Tuple[Grid, Grid]] = []
    seen_local: Set[str] = set()  # signatures canon_str pour cette série
    tries = 0
    max_tries = count * 10000  # assez large, vu les filtres de difficulté

    while len(puzzles) < count and tries < max_tries:
        if tries and tries % 500 == 0:
            print(f"[{profile.name}] tries={tries}, ok={len(puzzles)}/{count}")
        tries += 1

        puzzle, full = profile.generator()
        sig = canon_str(puzzle)
        h = hash_grid_sha256(puzzle)

        # 1. doublon local (dans cette série)
        if sig in seen_local:
            continue

        # 2. doublon global (dans tous les livres déjà générés)
        if h in global_hashes:
            continue

        # OK → puzzle accepté
        seen_local.add(sig)
        global_hashes.add(h)
        puzzles.append((puzzle, full))

    if len(puzzles) < count:
        raise RuntimeError(f"Seulement {len(puzzles)} puzzles générés pour le profil {profile.name}")

    # Sauvegarde de l'historique global mis à jour
    save_global_hashes(global_hashes)

    return puzzles