# sudoku_book.py
"""
Génération du PDF (puzzles + solutions) pour un ou plusieurs profils de difficulté.

Deux modes :
- build_book_pdf(...) : un seul niveau de difficulté pour tout le livre.
- build_book_pdf_with_ranges(...) : plages de numéros de puzzles avec difficultés différentes.
"""

from __future__ import annotations
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sudoku_core import Grid
from sudoku_difficulty import (
    DifficultyProfile,
    generate_puzzles_for_profile,
    hash_grid_sha256,
    book_hash_v1,
)

TRIM_W_DEFAULT = 6.0
TRIM_H_DEFAULT = 9.0

DEFAULT_GIVEN_COLOR = "black"
DEFAULT_ADDED_COLOR = "red"

BLOCK_SHADE_COLOR = "#e9e9e9"   # gris clair proche de l’exemple
BLOCK_SHADE_ALPHA = 1.0        # 1.0 = opaque



def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ---------- Dessin d'une grille puzzle ----------

def draw_puzzle_grid_at(ax, grid: Grid, left: float, bottom: float, size: float):
    cell = size / 9.0
    block = size / 3.0

    # --- Fond alterné par bloc 3x3 (style "échiquier" de blocs) ---
    for br in range(3):
        for bc in range(3):
            if (br + bc) % 2 == 0:
                ax.add_patch(
                    plt.Rectangle(
                        (left + bc * block, bottom + br * block),
                        block,
                        block,
                        facecolor=BLOCK_SHADE_COLOR,
                        edgecolor="none",
                        alpha=BLOCK_SHADE_ALPHA,
                        zorder=0,
                    )
                )

    # Cadre extérieur (au-dessus du fond)
    ax.add_patch(
        plt.Rectangle((left, bottom), size, size, fill=False, linewidth=3, color="k", zorder=3)
    )

    # Lignes internes (au-dessus du fond)
    for i in range(1, 9):
        lw = 2 if i % 3 == 0 else 0.8
        ax.plot(
            [left + i * cell, left + i * cell],
            [bottom, bottom + size],
            linewidth=lw,
            color="k",
            zorder=2,
        )
        ax.plot(
            [left, left + size],
            [bottom + i * cell, bottom + i * cell],
            linewidth=lw,
            color="k",
            zorder=2,
        )

    # Chiffres
    font_pts = cell * 0.5 * 72
    for r in range(9):
        for c in range(9):
            val = grid[r][c]
            if val != 0:
                x = left + c * cell + cell / 2
                y = bottom + (8 - r) * cell + cell * 0.47
                ax.text(
                    x,
                    y,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=font_pts,
                    fontweight="normal",
                    zorder=4,
                )



def draw_puzzles_page_figure(
    puzzles: List[Grid],
    trim_w: float,
    trim_h: float,
    rows: int,
    cols: int,
    page_num: int,
    title: str,
    puzzle_labels: Optional[List[str]] = None,  # labels optionnels (ex: "facile")
    start_idx: int = 1,  # index (1-based) du premier puzzle affiché sur cette page
):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(trim_w, trim_h))
    ax = plt.gca()
    ax.set_xlim(0, trim_w)
    ax.set_ylim(0, trim_h)
    ax.axis("off")

    margin_x = 0.5
    margin_y = 0.8
    avail_w = trim_w - 2 * margin_x
    avail_h = trim_h - 2 * margin_y

    cell_w = avail_w / cols
    cell_h = avail_h / rows
    size = min(cell_w, cell_h) * 0.90
    offset_x = (cell_w - size) / 2
    offset_y = (cell_h - size) / 2

    for idx, grid in enumerate(puzzles[: rows * cols]):
        r = idx // cols
        c = idx % cols

        left = margin_x + c * cell_w + offset_x
        bottom = margin_y + (rows - 1 - r) * cell_h + offset_y

        draw_puzzle_grid_at(ax, grid, left, bottom, size)

        # IMPORTANT :
        # La numérotation des puzzles ne doit PAS dépendre du numéro de page PDF global,
        # car le PDF peut contenir des pages d'intro, des couvertures, etc.
        # On utilise donc start_idx (1-based) + idx.
        puzzle_index = start_idx + idx

        label = ""
        if puzzle_labels is not None and idx < len(puzzle_labels):
            lab = (puzzle_labels[idx] or "").strip()
            if lab:
                label = f" — {lab}"

        # Numéro (et label éventuel) sous la grille
        ax.text(
            left + size / 2,
            bottom - 0.1,
            f"{puzzle_index}{label}",
            ha="center",
            va="top",
            fontsize=8,
        )

    ax.text(
        trim_w / 2,
        trim_h - 0.3,
        title,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    ax.text(
        trim_w - 0.2,
        0.2,
        str(page_num),
        ha="right",
        va="bottom",
        fontsize=10,
    )

    return fig


# ---------- Dessin des solutions miniatures ----------

def draw_sudoku_at(
    ax,
    solution_grid: Grid,
    left: float,
    bottom: float,
    size: float,
    puzzle_grid: Grid | None = None,
    given_color: str = DEFAULT_GIVEN_COLOR,
    added_color: str = DEFAULT_ADDED_COLOR,
):
    cell = size / 9.0
    block = size / 3.0

    # --- Fond alterné par bloc 3x3 ---
    for br in range(3):
        for bc in range(3):
            if (br + bc) % 2 == 0:
                ax.add_patch(
                    plt.Rectangle(
                        (left + bc * block, bottom + br * block),
                        block,
                        block,
                        facecolor=BLOCK_SHADE_COLOR,
                        edgecolor="none",
                        alpha=BLOCK_SHADE_ALPHA,
                        zorder=0,
                    )
                )

    ax.add_patch(
        plt.Rectangle((left, bottom), size, size, fill=False, linewidth=1.25, color="k", zorder=3)
    )

    for i in range(1, 9):
        lw = 0.6 if i % 3 == 0 else 0.25
        ax.plot(
            [left + i * cell, left + i * cell],
            [bottom, bottom + size],
            linewidth=lw,
            color="k",
            zorder=2,
        )
        ax.plot(
            [left, left + size],
            [bottom + i * cell, bottom + i * cell],
            linewidth=lw,
            color="k",
            zorder=2,
        )

    font_pts = cell * 0.5 * 72

    given = [[False] * 9 for _ in range(9)]
    if puzzle_grid is not None:
        for r in range(9):
            for c in range(9):
                if puzzle_grid[r][c] != 0:
                    given[r][c] = True

    for r in range(9):
        for c in range(9):
            v = solution_grid[r][c]
            if v:
                x = left + c * cell + cell / 2
                y = bottom + (8 - r) * cell + cell * 0.47

                if puzzle_grid is not None and not given[r][c]:
                    color = added_color
                    weight = "bold"
                else:
                    color = given_color
                    weight = "normal"

                ax.text(
                    x,
                    y,
                    str(v),
                    ha="center",
                    va="center",
                    fontsize=font_pts,
                    fontweight=weight,
                    color=color,
                    zorder=4,
                )



def draw_solutions_page_figure(
    puzzles_and_solutions: List[Tuple[Grid, Grid]],
    trim_w: float,
    trim_h: float,
    rows: int,
    cols: int,
    page_num: int,   # numéro de page PDF global (footer)
    start_idx: int,  # index (1-based) du premier puzzle affiché sur cette page de solutions
    given_color: str,
    added_color: str,
):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(trim_w, trim_h))
    ax = plt.gca()
    ax.set_xlim(0, trim_w)
    ax.set_ylim(0, trim_h)
    ax.axis("off")

    margin_x = 0.6
    margin_y = 0.95
    avail_w = trim_w - 2 * margin_x
    avail_h = trim_h - 2 * margin_y
    cell_w = avail_w / cols
    cell_h = avail_h / rows
    size = min(cell_w, cell_h) * 0.90
    offset_x = (cell_w - size) / 2
    offset_y = (cell_h - size) / 2

    for idx, (puz, sol) in enumerate(puzzles_and_solutions[: rows * cols]):
        r = idx // cols
        c = idx % cols
        left = margin_x + c * cell_w + offset_x
        bottom = margin_y + (rows - 1 - r) * cell_h + offset_y
        draw_sudoku_at(
            ax,
            sol,
            left,
            bottom,
            size,
            puzzle_grid=puz,
            given_color=given_color,
            added_color=added_color,
        )

        ax.text(
            left + size / 2,
            bottom - 0.1,
            f"{start_idx + idx}",
            ha="center",
            va="top",
            fontsize=8,
        )

    count = len(puzzles_and_solutions)
    first = start_idx
    last = first + count - 1
    title_str = f"Solutions {first}" if first == last else f"Solutions {first}–{last}"

    ax.text(
        trim_w / 2,
        trim_h - 0.3,
        title_str,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    ax.text(
        trim_w - 0.2,
        0.2,
        str(page_num),
        ha="right",
        va="bottom",
        fontsize=10,
    )

    return fig


# ---------- Helper interne : dessiner un livre à partir d'une liste de puzzles ----------

def _render_book_from_puzzles(
    puzzles: List[Tuple[Grid, Grid]],
    output_path: str,
    title: str,
    trim_w: float,
    trim_h: float,
    puzzles_per_page: int,
    puzzle_rows: int,
    puzzle_cols: int,
    solutions_per_page: int,
    solution_rows: int,
    solution_cols: int,
    given_color: str,
    added_color: str,
    puzzle_labels: Optional[List[str]] = None,  # labels par puzzle (facile/moyen/...)
) -> Tuple[List[str], str]:
    """
    Dessine les pages puzzles + solutions dans un PDF, à partir de la liste (puzzle, solution).
    Retourne (per_puzzle_hashes, book_hash).
    """
    page_no = 1

    with PdfPages(output_path) as pdf:
        # ---------- Pages intro (images) ----------
        import os
        import matplotlib.image as mpimg

        intro_dir = os.path.join(os.path.dirname(__file__), "pages_intro")
        intro_images = ["1.png", "2.png", "3.png", "4.png"]

        for img_name in intro_images:
            img_path = os.path.join(intro_dir, img_name)

            fig, ax = plt.subplots(figsize=(trim_w, trim_h))  # A4 en pouces
            ax.axis("off")

            img = mpimg.imread(img_path)

            # Si l'image est en niveaux de gris (matrice 2D), imshow applique une colormap par défaut (viridis)
            # -> il faut forcer l'affichage en gris.
            if img.ndim == 2:
                vmax = 255 if str(img.dtype) == "uint8" else 1
                ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, interpolation="none")
            else:
                # Image RGB/RGBA -> pas de colormap
                ax.imshow(img, interpolation="none")

            ax.set_aspect("auto")

            ax.text(
                0.98, 0.02,  # bas droite
                str(page_no),
                ha="right",
                va="bottom",
                fontsize=10,
                transform=ax.transAxes,  # <<< LA LIGNE IMPORTANTE
            )

            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)
            page_no += 1

        # ---------- Pages puzzles ----------
        puzzle_pages = list(chunk(puzzles, puzzles_per_page))

        labels_pages = None
        if puzzle_labels is not None:
            labels_pages = list(chunk(puzzle_labels, puzzles_per_page))

        for page_i, page_puzzles in enumerate(puzzle_pages):
            grids = [p for (p, _s) in page_puzzles]
            page_labels = None
            if labels_pages is not None and page_i < len(labels_pages):
                page_labels = labels_pages[page_i]

            # Index (1-based) du premier puzzle de cette page de puzzles,
            # indépendant du numéro de page PDF (intro/couverture/etc.)
            start_idx = page_i * puzzles_per_page + 1

            fig = draw_puzzles_page_figure(
                grids,
                trim_w=trim_w,
                trim_h=trim_h,
                rows=puzzle_rows,
                cols=puzzle_cols,
                page_num=page_no,
                title=title,
                puzzle_labels=page_labels,
                start_idx=start_idx,
            )
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)
            page_no += 1

        # ---------- Pages solutions ----------
        pages = list(chunk(puzzles, solutions_per_page))
        for sol_i, puz_sols in enumerate(pages, start=1):
            start_idx = (sol_i - 1) * solutions_per_page + 1

            fig = draw_solutions_page_figure(
                puz_sols,
                trim_w=trim_w,
                trim_h=trim_h,
                rows=solution_rows,
                cols=solution_cols,
                page_num=page_no,      # page PDF globale
                start_idx=start_idx,   # index puzzle (section solutions)
                given_color=given_color,
                added_color=added_color,
            )
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)
            page_no += 1

    per_puzzle_hashes = [hash_grid_sha256(p) for (p, _s) in puzzles]
    book_hash = book_hash_v1(puzzles)
    return per_puzzle_hashes, book_hash


# ---------- Mode 1 : un seul profil sur tout le livre ----------
PROFILE_NAME_FR = {"easy": "facile", "medium": "moyen", "hard": "difficile"}


def build_book_pdf(
    profile: DifficultyProfile,
    output_path: str,
    n_puzzles: int,
    title: str,
    authors: str,  # pas utilisé dans le dessin ici, mais gardé pour compat
    trim_w: float = TRIM_W_DEFAULT,
    trim_h: float = TRIM_H_DEFAULT,
    puzzles_per_page: int = 1,
    puzzle_rows: int = 1,
    puzzle_cols: int = 1,
    solutions_per_page: int = 9,
    solution_rows: int = 3,
    solution_cols: int = 3,
    given_color: str = DEFAULT_GIVEN_COLOR,
    added_color: str = DEFAULT_ADDED_COLOR,
) -> Tuple[List[Tuple[Grid, Grid]], List[str], str]:
    """
    Génère le PDF complet pour un profil de difficulté unique, et renvoie :
    - la liste (puzzle, solution)
    - la liste des hashs par puzzle
    - le hash global du "livre"

    ⚠️ Version SANS page de couverture.
    """
    puzzles = generate_puzzles_for_profile(profile, n_puzzles)

    label_fr = PROFILE_NAME_FR.get(profile.name, profile.name)
    puzzle_labels = [label_fr] * n_puzzles

    per_puzzle_hashes, book_hash = _render_book_from_puzzles(
        puzzles=puzzles,
        output_path=output_path,
        title=f"{title} — Niveau {label_fr}",
        trim_w=trim_w,
        trim_h=trim_h,
        puzzles_per_page=puzzles_per_page,
        puzzle_rows=puzzle_rows,
        puzzle_cols=puzzle_cols,
        solutions_per_page=solutions_per_page,
        solution_rows=solution_rows,
        solution_cols=solution_cols,
        given_color=given_color,
        added_color=added_color,
        puzzle_labels=puzzle_labels,
    )

    return puzzles, per_puzzle_hashes, book_hash


# ---------- Mode 2 : plages de numéros avec difficultés différentes ----------

def build_book_pdf_with_ranges(
    range_specs: List[Tuple[int, int, DifficultyProfile]],
    output_path: str,
    title: str,
    authors: str,
    trim_w: float = TRIM_W_DEFAULT,
    trim_h: float = TRIM_H_DEFAULT,
    puzzles_per_page: int = 1,
    puzzle_rows: int = 1,
    puzzle_cols: int = 1,
    solutions_per_page: int = 9,
    solution_rows: int = 3,
    solution_cols: int = 3,
    given_color: str = DEFAULT_GIVEN_COLOR,
    added_color: str = DEFAULT_ADDED_COLOR,
) -> Tuple[List[Tuple[Grid, Grid]], List[str], str]:
    """
    Génère un livre où certaines plages de numéros de puzzles
    ont des difficultés différentes.

    range_specs = [
      (start_index, end_index, profile),
      ...
    ]
    """
    range_specs = sorted(range_specs, key=lambda x: x[0])

    puzzles: List[Tuple[Grid, Grid]] = []
    puzzle_labels: List[str] = []

    for start_idx, end_idx, profile in range_specs:
        if end_idx < start_idx:
            raise ValueError(f"Plage invalide: {start_idx}–{end_idx}")
        count = end_idx - start_idx + 1
        print(f"Génération {count} puzzle(s) pour {profile.name} (puzzles {start_idx}–{end_idx})")

        part = generate_puzzles_for_profile(profile, count)
        puzzles.extend(part)

        label_fr = PROFILE_NAME_FR.get(profile.name, profile.name)
        puzzle_labels.extend([label_fr] * count)

    render_title = f"{title} — Mode mix"

    per_puzzle_hashes, book_hash = _render_book_from_puzzles(
        puzzles=puzzles,
        output_path=output_path,
        title=render_title,
        trim_w=trim_w,
        trim_h=trim_h,
        puzzles_per_page=puzzles_per_page,
        puzzle_rows=puzzle_rows,
        puzzle_cols=puzzle_cols,
        solutions_per_page=solutions_per_page,
        solution_rows=solution_rows,
        solution_cols=solution_cols,
        given_color=given_color,
        added_color=added_color,
        puzzle_labels=puzzle_labels,
    )

    return puzzles, per_puzzle_hashes, book_hash
