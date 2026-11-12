"""
Limpeza do dicionario_MADLAD_10b.json, feita uma vez. Três correções, nesta
ordem:

1. EXCLUSÃO: remove verbetes em que o lado tca é um morfema preso/partícula
   e o lado es/pt só explica a gramática ("indica plural", "se usa para
   unir sustantivos...") em vez de traduzir. Isso ensinaria errado o modelo.
   Os verbetes "Indica una especie de X" (identificação de espécie) NÃO são
   excluídos, porque muitas vezes são a única tradução disponível para
   plantas e animais sem nome comum em pt/es.

2. REMOÇÃO DE PARÊNTESES: quando o texto é uma tradução de verdade seguida
   de uma nota entre parênteses (ex.: "bijao (hojas grandes que se utilizan
   para envolver...)"), tira a nota e deixa só o par palavra-palavra. Essas
   notas de dicionário não são uma tradução natural.

3. DUPLICADOS: remove linhas repetidas (mesmo tca+es+pt com outro id).

Grava o JSON limpo e um relatório em texto com tudo que mudou.

Executar:
    python clean_dictionary.py
"""

import json
import re
from pathlib import Path

INPUT_PATH = Path("./raw_data/dicionario_MADLAD_10b.json")
OUTPUT_PATH = Path("./raw_data/dicionario_MADLAD_10b_cleaned.json")
REPORT_PATH = Path("./work/dictionary_cleaning_report.txt")

EXPLANATION_MARKERS = [
    "se usa", "se refiere", "indica", "se emplea", "denota",
    "sirve para", "se dice", "se utiliza", "expresa", "se añade",
]


def word_count(s):
    return len(s.split())


def is_headword(record):
    return bool(record.get("type", "").strip())


def has_explanation_marker(text):
    lower = text.lower()
    return any(m in lower for m in EXPLANATION_MARKERS)


def is_bound_morpheme(tca_text):
    t = tca_text.strip()
    return t.startswith("-") or t.endswith("-")


def is_species_id(es_text):
    return "especie de" in es_text.lower()


def classify(record):
    """Retorna 'hard_exclude', 'species_keep', 'gloss_strip' ou 'keep'."""
    if not is_headword(record):
        return "keep"  # frases de exemplo não são alteradas

    tca_words = word_count(record["tca"])
    es_words = word_count(record["es"])
    pt_words = word_count(record["pt"])
    has_parens = "(" in record["es"] and ")" in record["es"]

    looks_like_explanation = tca_words <= 2 and has_explanation_marker(record["es"])

    if looks_like_explanation:
        # Os parênteses precisam ser checados ANTES de concluir que é só uma
        # explicação. Expressões como "se usa" muitas vezes aparecem dentro de uma
        # nota sobre como o OBJETO é usado ("madeira que se usa para canoas"), e não
        # sobre o uso gramatical da palavra. Se há uma palavra antes dos parênteses
        # (ex.: "aura" -> "añuje caspi (árbol cuya madera se usa para canoas)"), é
        # uma tradução com nota e não pode ser apagada.
        if has_parens:
            return "gloss_strip"
        if is_species_id(record["es"]):
            return "species_keep"
        return "hard_exclude"

    # Formato de glosa: palavra curta, es/pt longo, com nota entre parênteses,
    # e que não caiu na checagem de explicação acima.
    is_short_headword_long_target = tca_words <= 2 and (es_words >= 6 or pt_words >= 6)
    if is_short_headword_long_target and has_parens:
        return "gloss_strip"

    return "keep"


def strip_parenthetical(text):
    # Remove qualquer trecho (...), junto com o espaço antes dele, e depois
    # limpa espaços duplos e pontuação solta antes do ponto final.
    cleaned = re.sub(r"\s*\([^)]*\)", "", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
    return cleaned.strip(" ,")


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"{len(data)} verbetes carregados de {INPUT_PATH}")

    excluded = []
    gloss_stripped = []
    species_kept = []
    cleaned_records = []
    seen_full_rows = {}
    exact_duplicates = []

    for r in data:
        category = classify(r)

        if category == "hard_exclude":
            excluded.append(r)
            continue  # removido

        record = dict(r)  # trabalha numa cópia

        if category == "gloss_strip":
            before_es, before_pt = record["es"], record["pt"]
            record["es"] = strip_parenthetical(record["es"])
            record["pt"] = strip_parenthetical(record["pt"])
            gloss_stripped.append((r["id"], before_es, record["es"], before_pt, record["pt"]))

        if category == "species_keep":
            species_kept.append(r["id"])

        # Checagem de duplicados (depois da limpeza, para pegar também as glosas
        # que ficaram iguais a outra linha)
        key = (record["tca"].strip(), record["es"].strip(), record["pt"].strip())
        if key in seen_full_rows:
            exact_duplicates.append((r["id"], seen_full_rows[key]))
            continue  # removido por ser duplicado

        seen_full_rows[key] = r["id"]
        cleaned_records.append(record)

    Path("./raw_data").mkdir(parents=True, exist_ok=True)
    Path("./work").mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)

    # ---- Relatório ----
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Relatório de limpeza do dicionário\n{'='*60}\n\n")
        f.write(f"Entrada: {len(data)} verbetes\n")
        f.write(f"Saída:   {len(cleaned_records)} verbetes\n\n")

        f.write(f"1. EXCLUÍDOS (só explicação gramatical, sem tradução): "
                f"{len(excluded)}\n" + "-"*60 + "\n")
        for r in excluded:
            f.write(f"  id={r['id']:>5}  tca={r['tca']!r:20}  es={r['es']!r}\n")

        f.write(f"\n2. IDENTIFICAÇÃO DE ESPÉCIE MANTIDA (não excluída): "
                f"{len(species_kept)}\n" + "-"*60 + "\n")
        f.write(f"  ids: {species_kept}\n")

        f.write(f"\n3. PARÊNTESES REMOVIDOS (tradução mantida, nota removida): "
                f"{len(gloss_stripped)}\n" + "-"*60 + "\n")
        for rid, before_es, after_es, before_pt, after_pt in gloss_stripped:
            f.write(f"  id={rid}\n")
            f.write(f"    es: {before_es!r} -> {after_es!r}\n")
            f.write(f"    pt: {before_pt!r} -> {after_pt!r}\n")

        f.write(f"\n4. DUPLICADOS REMOVIDOS (mantida a primeira ocorrência): "
                f"{len(exact_duplicates)}\n" + "-"*60 + "\n")
        for dropped_id, kept_id in exact_duplicates:
            f.write(f"  id={dropped_id} duplicado do id={kept_id}\n")

    print(f"\nHard-excluded (dropped): {len(excluded)}")
    print(f"Identificação de espécie mantida: {len(species_kept)}")
    print(f"Parênteses removidos: {len(gloss_stripped)}")
    print(f"Duplicados removidos: {len(exact_duplicates)}")
    print(f"\nArquivo limpo: {OUTPUT_PATH} ({len(cleaned_records)} verbetes)")
    print(f"Relatório completo: {REPORT_PATH}")


if __name__ == "__main__":
    main()
