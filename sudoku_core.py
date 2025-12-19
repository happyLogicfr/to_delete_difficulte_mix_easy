# sudoku_core.py
"""
Moteur Sudoku commun :
- grille 9x9
- backtracking
- calcul de candidats
- UNITS / PEERS
"""

from __future__ import annotations
import random
from copy import deepcopy
from typing import List, Tuple, Dict, Set

Grid = List[List[int]]
Pos = Tuple[int, int]

# ---------- UNITS & PEERS communs ----------

UNITS: List[List[Pos]] = []
PEERS: Dict[Pos, Set[Pos]] = {}

# Lignes
for r in range(9):
    UNITS.append([(r, c) for c in range(9)])
# Colonnes
for c in range(9):
    UNITS.append([(r, c) for r in range(9)])
# Blocs 3x3
for br in range(0, 9, 3):
    for bc in range(0, 9, 3):
        UNITS.append([(br + dr, bc + dc) for dr in range(3) for dc in range(3)])

# Voisins de chaque case
for r in range(9):
    for c in range(9):
        peers = set()
        peers |= {(r, cc) for cc in range(9) if cc != c}
        peers |= {(rr, c) for rr in range(9) if rr != r}
        br, bc = 3 * (r // 3), 3 * (c // 3)
        peers |= {
            (br + dr, bc + dc)
            for dr in range(3)
            for dc in range(3)
            if (br + dr, bc + dc) != (r, c)
        }
        PEERS[(r, c)] = peers


# ---------- Backtracking / unicité ----------

def _find_empty(grid: Grid) -> Pos | None:
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return r, c
    return None


def _candidates(grid: Grid, r: int, c: int) -> List[int]:
    used = set(grid[r]) | {grid[i][c] for i in range(9)}
    br, bc = 3 * (r // 3), 3 * (c // 3)
    used |= {
        grid[i][j]
        for i in range(br, br + 3)
        for j in range(bc, bc + 3)
    }
    return [v for v in range(1, 10) if v not in used]


def count_solutions(grid: Grid, limit: int = 2) -> int:
    """
    Compte les solutions de la grille (backtracking),
    s'arrête dès qu'on atteint 'limit'.
    """
    empty = _find_empty(grid)
    if not empty:
        return 1
    r, c = empty
    sols = 0
    for v in _candidates(grid, r, c):
        grid[r][c] = v
        sols += count_solutions(grid, limit)
        grid[r][c] = 0
        if sols >= limit:
            break
    return sols


def has_unique_solution(grid: Grid) -> bool:
    return count_solutions(deepcopy(grid), limit=2) == 1


def generate_full_grid() -> Grid:
    """Génère une grille complète valide (9x9)."""
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
            if all(grid[r][x] != v for x in range(9)) and all(
                grid[x][c] != v for x in range(9)
            ):
                br, bc = 3 * (r // 3), 3 * (c // 3)
                if all(
                    grid[rr][cc] != v
                    for rr in range(br, br + 3)
                    for cc in range(bc, bc + 3)
                ):
                    grid[r][c] = v
                    if backtrack(nr, nc):
                        return True
                    grid[r][c] = 0
        return False

    backtrack()
    return grid


def grid_candidates(grid: Grid) -> Dict[Pos, Set[int]]:
    """Retourne un dict {(r,c): {candidats}} pour les cellules vides."""
    cands: Dict[Pos, Set[int]] = {}
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                used = set(grid[r]) | {grid[i][c] for i in range(9)}
                br, bc = 3 * (r // 3), 3 * (c // 3)
                used |= {
                    grid[i][j]
                    for i in range(br, br + 3)
                    for j in range(bc, bc + 3)
                }
                cands[(r, c)] = {v for v in range(1, 10) if v not in used}
    return cands
