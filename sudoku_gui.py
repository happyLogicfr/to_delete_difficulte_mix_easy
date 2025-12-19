# sudoku_gui.py
"""
Interface CustomTkinter pour générer un livre de Sudoku
avec plusieurs difficultés (facile+ / facile / facile- / moyen / difficile)
et un mode mixte par plages de numéros.
"""

from __future__ import annotations
import math
import os

import customtkinter as ctk
from tkinter import messagebox

from sudoku_difficulty import PROFILES, DifficultyProfile
from sudoku_book import (
    build_book_pdf,
    build_book_pdf_with_ranges,
    TRIM_W_DEFAULT,
    TRIM_H_DEFAULT,
    DEFAULT_GIVEN_COLOR,
    DEFAULT_ADDED_COLOR,
)

# ---------------------------
# Libellés FR pour l'UI
# ---------------------------
DIFF_KEY_TO_LABEL_FR = {
    "easy_plus": "facile +",
    "easy": "facile",
    "easy_minus": "facile −",
    "medium": "moyen",
    "hard": "difficile",
}
DIFF_LABEL_FR_TO_KEY = {v: k for k, v in DIFF_KEY_TO_LABEL_FR.items()}


def diff_key_to_label_fr(key: str) -> str:
    return DIFF_KEY_TO_LABEL_FR.get(key, key)


def diff_label_fr_to_key(label: str) -> str:
    return DIFF_LABEL_FR_TO_KEY.get(label, label)


# Config par défaut
DEFAULT_N_PUZZLES = 20
DEFAULT_BOOK_TITLE = "Sudoku"
DEFAULT_BOOK_AUTHORS = "Quentin et Thibault"


def launch_gui():
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("Générateur de livre Sudoku")

    # Variables TK
    n_puzzles_var = ctk.StringVar(value=str(DEFAULT_N_PUZZLES))
    title_var = ctk.StringVar(value=DEFAULT_BOOK_TITLE)
    authors_var = ctk.StringVar(value=DEFAULT_BOOK_AUTHORS)
    trim_w_var = ctk.StringVar(value=str(TRIM_W_DEFAULT))
    trim_h_var = ctk.StringVar(value=str(TRIM_H_DEFAULT))
    given_color_var = ctk.StringVar(value=DEFAULT_GIVEN_COLOR)
    added_color_var = ctk.StringVar(value=DEFAULT_ADDED_COLOR)
    sol_rows_var = ctk.StringVar(value="3")
    sol_cols_var = ctk.StringVar(value="3")
    puzzles_per_page_var = ctk.StringVar(value="1")  # 1,2,3,4

    # Difficulté : profils + mode mix
    # -> on stocke le libellé FR dans la var, et on convertit en clé EN au moment d'utiliser PROFILES
    difficulty_var = ctk.StringVar(value=DIFF_KEY_TO_LABEL_FR["easy"])  # "facile +", "facile", "facile −", "moyen", "difficile", "mix"
    status_var = ctk.StringVar(value="Prêt.")

    app.grid_columnconfigure(0, weight=1)
    app.grid_columnconfigure(1, weight=1)

    # ----- Frame gauche -----
    frame_left = ctk.CTkFrame(app)
    frame_left.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    frame_left.grid_columnconfigure(1, weight=1)

    ctk.CTkLabel(
        frame_left,
        text="Paramètres généraux",
        font=ctk.CTkFont(size=16, weight="bold"),
    ).grid(row=0, column=0, columnspan=2, pady=(10, 20))

    ctk.CTkLabel(frame_left, text="Nombre de puzzles").grid(
        row=1, column=0, sticky="w", padx=5, pady=5
    )
    entry_n_puzzles = ctk.CTkEntry(frame_left, textvariable=n_puzzles_var)
    entry_n_puzzles.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

    ctk.CTkLabel(frame_left, text="Titre du livre").grid(
        row=2, column=0, sticky="w", padx=5, pady=5
    )
    ctk.CTkEntry(frame_left, textvariable=title_var).grid(
        row=2, column=1, sticky="ew", padx=5, pady=5
    )

    ctk.CTkLabel(frame_left, text="Auteurs").grid(
        row=3, column=0, sticky="w", padx=5, pady=5
    )
    ctk.CTkEntry(frame_left, textvariable=authors_var).grid(
        row=3, column=1, sticky="ew", padx=5, pady=5
    )

    # Difficulté ou mode mix
    ctk.CTkLabel(frame_left, text="Mode difficulté").grid(
        row=4, column=0, sticky="w", padx=5, pady=5
    )
    ctk.CTkOptionMenu(
        frame_left,
        # profils existants + entrée spéciale "mix"
        values=[diff_key_to_label_fr(k) for k in PROFILES.keys()] + ["mix"],
        variable=difficulty_var,
    ).grid(row=4, column=1, sticky="ew", padx=5, pady=5)

    # ------- Frame pour les plages (mode mix) -------
    frame_ranges = ctk.CTkFrame(frame_left)
    frame_ranges.grid(row=5, column=0, columnspan=2, padx=5, pady=(10, 5), sticky="ew")
    frame_ranges.grid_columnconfigure(3, weight=1)

    ctk.CTkLabel(
        frame_ranges,
        text="Plages de numéros (mode mix)",
        font=ctk.CTkFont(size=13, weight="bold"),
    ).grid(row=0, column=0, columnspan=4, pady=(5, 10), sticky="w")

    # Trois plages possibles : (de, à, difficulté)
    range1_from_var = ctk.StringVar(value="")
    range1_to_var = ctk.StringVar(value="")
    range1_diff_var = ctk.StringVar(value=DIFF_KEY_TO_LABEL_FR["easy"])

    range2_from_var = ctk.StringVar(value="")
    range2_to_var = ctk.StringVar(value="")
    range2_diff_var = ctk.StringVar(value=DIFF_KEY_TO_LABEL_FR["medium"])

    range3_from_var = ctk.StringVar(value="")
    range3_to_var = ctk.StringVar(value="")
    range3_diff_var = ctk.StringVar(value=DIFF_KEY_TO_LABEL_FR["hard"])

    def add_range_row(row, label, from_var, to_var, diff_var, default_diff_key):
        ctk.CTkLabel(frame_ranges, text=label).grid(
            row=row, column=0, sticky="w", padx=5, pady=2
        )
        ctk.CTkEntry(frame_ranges, width=40, textvariable=from_var).grid(
            row=row, column=1, sticky="ew", padx=5, pady=2
        )
        ctk.CTkLabel(frame_ranges, text="à").grid(
            row=row, column=2, sticky="w", padx=5, pady=2
        )
        ctk.CTkEntry(frame_ranges, width=40, textvariable=to_var).grid(
            row=row, column=3, sticky="ew", padx=5, pady=2
        )
        ctk.CTkOptionMenu(
            frame_ranges,
            values=[diff_key_to_label_fr(k) for k in PROFILES.keys()],
            variable=diff_var,
        ).grid(row=row, column=4, sticky="ew", padx=5, pady=2)

        # valeur par défaut en FR
        diff_var.set(diff_key_to_label_fr(default_diff_key))

    add_range_row(1, "Plage 1 (puzzles)", range1_from_var, range1_to_var, range1_diff_var, "easy")
    add_range_row(2, "Plage 2 (puzzles)", range2_from_var, range2_to_var, range2_diff_var, "medium")
    add_range_row(3, "Plage 3 (puzzles)", range3_from_var, range3_to_var, range3_diff_var, "hard")

    ctk.CTkLabel(
        frame_ranges,
        text="Laisse une plage vide pour l'ignorer.\n"
             "Exemple : 1–10 facile, 11–20 moyen, 21–30 difficile.",
        justify="left",
    ).grid(row=4, column=0, columnspan=5, sticky="w", padx=5, pady=(5, 8))

    # ----- Sudoku par page -----
    ctk.CTkLabel(frame_left, text="Sudokus par page (puzzles)").grid(
        row=6, column=0, sticky="w", padx=5, pady=5
    )
    ctk.CTkOptionMenu(
        frame_left,
        values=["1", "2", "3", "4"],
        variable=puzzles_per_page_var,
    ).grid(row=6, column=1, sticky="ew", padx=5, pady=5)

    # ----- Frame droite -----
    frame_right = ctk.CTkFrame(app)
    frame_right.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    frame_right.grid_columnconfigure(1, weight=1)

    ctk.CTkLabel(
        frame_right,
        text="Mise en page & couleurs",
        font=ctk.CTkFont(size=16, weight="bold"),
    ).grid(row=0, column=0, columnspan=2, pady=(10, 20))

    ctk.CTkLabel(frame_right, text="Largeur page (TRIM_W)").grid(
        row=1, column=0, sticky="w", padx=5, pady=5
    )
    ctk.CTkEntry(frame_right, textvariable=trim_w_var).grid(
        row=1, column=1, sticky="ew", padx=5, pady=5
    )

    ctk.CTkLabel(frame_right, text="Hauteur page (TRIM_H)").grid(
        row=2, column=0, sticky="w", padx=5, pady=5
    )
    ctk.CTkEntry(frame_right, textvariable=trim_h_var).grid(
        row=2, column=1, sticky="ew", padx=5, pady=5
    )

    ctk.CTkLabel(frame_right, text="Couleur indices (GIVEN_COLOR)").grid(
        row=3, column=0, sticky="w", padx=5, pady=5
    )
    ctk.CTkEntry(frame_right, textvariable=given_color_var).grid(
        row=3, column=1, sticky="ew", padx=5, pady=5
    )

    ctk.CTkLabel(
        frame_right,
        text="Couleur solutions ajoutées (ADDED_COLOR)",
    ).grid(row=4, column=0, sticky="w", padx=5, pady=5)
    ctk.CTkEntry(frame_right, textvariable=added_color_var).grid(
        row=4, column=1, sticky="ew", padx=5, pady=5
    )

    ctk.CTkLabel(frame_right, text="Lignes solutions / page").grid(
        row=5, column=0, sticky="w", padx=5, pady=5
    )
    ctk.CTkEntry(frame_right, textvariable=sol_rows_var).grid(
        row=5, column=1, sticky="ew", padx=5, pady=5
    )

    ctk.CTkLabel(frame_right, text="Colonnes solutions / page").grid(
        row=6, column=0, sticky="w", padx=5, pady=5
    )
    ctk.CTkEntry(frame_right, textvariable=sol_cols_var).grid(
        row=6, column=1, sticky="ew", padx=5, pady=5
    )

    # ----- Bas -----
    frame_bottom = ctk.CTkFrame(app)
    frame_bottom.grid(
        row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew"
    )
    frame_bottom.grid_columnconfigure(0, weight=1)

    status_label = ctk.CTkLabel(frame_bottom, textvariable=status_var, anchor="w")
    status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

    generate_button = ctk.CTkButton(
        frame_bottom, text="Générer le PDF", command=lambda: on_generate()
    )
    generate_button.grid(row=0, column=1, padx=10, pady=5, sticky="e")

    # --- Affichage conditionnel du bloc "plages" ---

    def update_ranges_visibility(*args):
        mode = difficulty_var.get()
        if mode == "mix":
            frame_ranges.grid()     # montrer
        else:
            frame_ranges.grid_remove()  # cacher

    difficulty_var.trace_add("write", update_ranges_visibility)
    update_ranges_visibility()

    # ==========================
    #   ACTION : GÉNÉRER
    # ==========================

    def parse_ranges():
        """
        Lit les 3 lignes de plages et renvoie une liste:
          [(start, end, profile), ...]
        en ignorant les lignes vides.
        """
        ranges = []

        def add_if_valid(from_var, to_var, diff_var):
            s = from_var.get().strip()
            e = to_var.get().strip()
            if not s and not e:
                return
            if not s or not e:
                raise ValueError("Chaque plage doit avoir un début ET une fin.")
            try:
                start = int(s)
                end = int(e)
            except ValueError:
                raise ValueError("Les limites de plage doivent être des entiers.")
            if start <= 0 or end <= 0:
                raise ValueError("Les numéros de puzzle doivent être ≥ 1.")
            if end < start:
                raise ValueError(f"Plage invalide : {start}–{end} (fin < début).")

            diff_label = diff_var.get()
            diff_key = diff_label_fr_to_key(diff_label)

            if diff_key not in PROFILES:
                raise ValueError(f"Difficulté inconnue dans une plage : {diff_label}")

            profile = PROFILES[diff_key]
            ranges.append((start, end, profile))

        add_if_valid(range1_from_var, range1_to_var, range1_diff_var)
        add_if_valid(range2_from_var, range2_to_var, range2_diff_var)
        add_if_valid(range3_from_var, range3_to_var, range3_diff_var)

        if not ranges:
            raise ValueError("En mode mix, tu dois définir au moins une plage.")

        # Tri + vérif chevauchements
        ranges.sort(key=lambda x: x[0])
        for i in range(len(ranges) - 1):
            s1, e1, _ = ranges[i]
            s2, e2, _ = ranges[i + 1]
            if e1 >= s2:
                raise ValueError(f"Les plages se chevauchent : {s1}–{e1} et {s2}–{e2}.")

        return ranges

    def on_generate():
        try:
            book_title = title_var.get().strip() or DEFAULT_BOOK_TITLE
            book_authors = authors_var.get().strip() or DEFAULT_BOOK_AUTHORS

            trim_w = float(trim_w_var.get())
            trim_h = float(trim_h_var.get())
            if trim_w <= 0 or trim_h <= 0:
                raise ValueError("TRIM_W et TRIM_H doivent être > 0.")

            given_color = given_color_var.get().strip() or DEFAULT_GIVEN_COLOR
            added_color = added_color_var.get().strip() or DEFAULT_ADDED_COLOR

            sol_rows = int(sol_rows_var.get())
            sol_cols = int(sol_cols_var.get())
            if sol_rows <= 0 or sol_cols <= 0:
                raise ValueError("Les lignes/colonnes de solutions doivent être > 0.")
            solutions_per_page = sol_rows * sol_cols

            puzzles_per_page = int(puzzles_per_page_var.get())
            if puzzles_per_page not in (1, 2, 3, 4):
                raise ValueError("Sudokus par page doit être 1, 2, 3 ou 4.")

            # Layout automatique
            if puzzles_per_page == 1:
                puzzle_rows, puzzle_cols = 1, 1
            elif puzzles_per_page == 2:
                puzzle_rows, puzzle_cols = 2, 1
            elif puzzles_per_page == 3:
                puzzle_rows, puzzle_cols = 3, 1
            else:  # 4
                puzzle_rows, puzzle_cols = 2, 2

            mode_label = difficulty_var.get()

            # --- Mode simple : une seule difficulté pour tout le livre ---
            if mode_label != "mix":
                mode_key = diff_label_fr_to_key(mode_label)
                if mode_key not in PROFILES:
                    raise ValueError(f"Difficulté inconnue : {mode_label}")

                profile: DifficultyProfile = PROFILES[mode_key]

                n_puzzles = int(n_puzzles_var.get())
                if n_puzzles <= 0:
                    raise ValueError("Le nombre de puzzles doit être > 0.")

                output_file = f"sudoku_book_{n_puzzles}_puzzles_{profile.name}.pdf"

                status_var.set("Génération du PDF en cours (mode simple)...")
                app.update_idletasks()

                puzzle_pages = math.ceil(n_puzzles / puzzles_per_page)
                solution_pages = math.ceil(n_puzzles / solutions_per_page)
                total_pages = puzzle_pages + solution_pages  # plus de couverture

                print(f"Génération du PDF : {output_file}")
                print(f"- Difficulté : {diff_key_to_label_fr(profile.name)}")
                print(
                    f"- {n_puzzles} puzzles ({puzzles_per_page} par page → ~{puzzle_pages} page(s))"
                )
                print(
                    f"- {n_puzzles} solutions (~{solution_pages} page(s), {solutions_per_page} par page)"
                )
                print(f"- Total pages estimées : {total_pages}")

                puzzles, per_puzzle_hashes, book_hash = build_book_pdf(
                    profile=profile,
                    output_path=output_file,
                    n_puzzles=n_puzzles,
                    title=book_title,
                    authors=book_authors,
                    trim_w=trim_w,
                    trim_h=trim_h,
                    puzzles_per_page=puzzles_per_page,
                    puzzle_rows=puzzle_rows,
                    puzzle_cols=puzzle_cols,
                    solutions_per_page=solutions_per_page,
                    solution_rows=sol_rows,
                    solution_cols=sol_cols,
                    given_color=given_color,
                    added_color=added_color,
                )

            # --- Mode mix : plages de numéros + profils différents ---
            else:
                ranges = parse_ranges()
                # nombre total de puzzles = max de tous les "end"
                n_puzzles = max(end for (start, end, _p) in ranges)
                n_puzzles_var.set(str(n_puzzles))  # maj de l'affichage

                output_file = f"sudoku_book_{n_puzzles}_puzzles_mix.pdf"

                status_var.set("Génération du PDF en cours (mode mix)...")
                app.update_idletasks()

                puzzle_pages = math.ceil(n_puzzles / puzzles_per_page)
                solution_pages = math.ceil(n_puzzles / solutions_per_page)
                total_pages = puzzle_pages + solution_pages

                print(f"Génération du PDF : {output_file}")
                print(f"- Mode mix avec plages :")
                for (start, end, prof) in ranges:
                    print(f"  - Puzzles {start}–{end} : {diff_key_to_label_fr(prof.name)}")
                print(
                    f"- {n_puzzles} puzzles ({puzzles_per_page} par page → ~{puzzle_pages} page(s))"
                )
                print(
                    f"- {n_puzzles} solutions (~{solution_pages} page(s), {solutions_per_page} par page)"
                )
                print(f"- Total pages estimées : {total_pages}")

                puzzles, per_puzzle_hashes, book_hash = build_book_pdf_with_ranges(
                    range_specs=ranges,
                    output_path=output_file,
                    title=book_title,
                    authors=book_authors,
                    trim_w=trim_w,
                    trim_h=trim_h,
                    puzzles_per_page=puzzles_per_page,
                    puzzle_rows=puzzle_rows,
                    puzzle_cols=puzzle_cols,
                    solutions_per_page=solutions_per_page,
                    solution_rows=sol_rows,
                    solution_cols=sol_cols,
                    given_color=given_color,
                    added_color=added_color,
                )

            # Résumé hash & statut
            short_hashes = [h[:8] for h in per_puzzle_hashes]
            print("🔑 Hashs puzzles :", short_hashes)
            print("📘 Hash du livre (ordre-agnostique) :", book_hash)

            status_var.set(f"✅ PDF généré : {output_file}")
            abs_path = os.path.abspath(output_file)
            messagebox.showinfo("Terminé", f"PDF généré :\n{abs_path}")

        except Exception as e:
            status_var.set("❌ Erreur lors de la génération.")
            messagebox.showerror("Erreur", f"Une erreur est survenue : {e}")

    # remplacer la commande par la version interne
    generate_button.configure(command=on_generate)

    app.mainloop()


if __name__ == "__main__":
    launch_gui()
