# sudoku_hash_db.py
"""
Gestion d'un fichier global pour stocker tous les hashes
des puzzles déjà utilisés (toutes difficultés confondues).

Format : un hash SHA256 par ligne.
"""

import os
from typing import Set

# Fichier global
HASH_DB_FILE = "puzzle_hashes_all.txt"


# Charger l'ensemble des hashes déjà utilisés
def load_global_hashes(path: str = HASH_DB_FILE) -> Set[str]:
    hashes = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                h = line.strip()
                if h:
                    hashes.add(h)
    return hashes


# Sauvegarder l'ensemble des hashes utilisés
def save_global_hashes(hashes: Set[str], path: str = HASH_DB_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for h in sorted(hashes):
            f.write(h + "\n")
