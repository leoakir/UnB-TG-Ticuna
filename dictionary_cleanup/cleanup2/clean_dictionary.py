"""
Limpeza do dicionario_MADLAD_10b.json, feita uma vez. Correções, nesta ordem:

1. EXCLUSÃO: remove verbetes em que o lado tca é um morfema preso/partícula
   e o lado es/pt só explica a gramática ("indica plural", "se usa para
   unir sustantivos...") em vez de traduzir. Isso ensinaria errado o modelo.
   Os verbetes "Indica una especie de X" (identificação de espécie) NÃO são
   excluídos, porque muitas vezes são a única tradução disponível para
   plantas e animais sem nome comum em pt/es.

2. REMOÇÃO DE PARÊNTESES: quando o texto é uma tradução de verdade seguida
   de uma nota entre parênteses (ex.: "bijao (hojas grandes que se utilizan
   para envolver...)"), tira a nota e deixa só o par palavra-palavra.

3. SINÔNIMOS: listas de sinônimos separadas por vírgula no campo pt viram
   uma linha por sinônimo.

4. DUPLICADOS: remove linhas repetidas (mesmo tca+es+pt com outro id).

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

SYNONYM_SPLIT_MAX_WORDS = 4  # cada trecho entre vírgulas precisa ter no máximo este
                              # número de palavras para contar como sinônimo, e não
                              # como pedaço de uma frase maior


def word_count(s):
    return len(s.split())


def is_headword_like(tca_text):
    """Detecta verbetes pelo conteúdo, e não pelo campo 'type': ~1.305 verbetes
    reais têm esse campo vazio (ele não é confiável nos dados de origem), e
    filtrar só por ele deixava esses verbetes de fora sem aviso. Um tca curto
    que não termina com pontuação de frase é um sinal bem mais confiável."""
    t = tca_text.strip()
    return len(t) > 0 and t[-1] not in ".!?" and word_count(t) <= 4


def has_explanation_marker(text):
    lower = text.lower()
    return any(m in lower for m in EXPLANATION_MARKERS)


def is_species_id(es_text):
    return "especie de" in es_text.lower()


def strip_parenthetical(text):
    cleaned = re.sub(r"\s*\([^)]*\)", "", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.;])", r"\1", cleaned)
    return cleaned.strip(" ,")


def classify(record):
    """Retorna 'hard_exclude', 'species_keep', 'gloss_strip' ou 'keep'."""
    if not is_headword_like(record["tca"]):
        return "keep"  # frases não são alteradas

    es_words = word_count(record["es"])
    pt_words = word_count(record["pt"])
    has_parens = "(" in record["es"] and ")" in record["es"]

    looks_like_explanation = has_explanation_marker(record["es"])

    if looks_like_explanation:
        if has_parens:
            return "gloss_strip"
        if is_species_id(record["es"]):
            return "species_keep"
        return "hard_exclude"

    is_long_target = es_words >= 6 or pt_words >= 6
    if is_long_target and has_parens:
        return "gloss_strip"

    return "keep"


def synonym_segments(text, max_words=SYNONYM_SPLIT_MAX_WORDS):
    """Separa um campo com vírgulas em sinônimos, mas só se TODOS os trechos
    (sem as notas entre parênteses) forem curtos. Um trecho longo indica uma
    frase descritiva que por acaso tem vírgula, e não uma lista de sinônimos;
    nesse caso o campo não é separado."""
    raw_segs = [s.strip() for s in text.split(",") if s.strip()]
    if len(raw_segs) < 2:
        return None
    stripped = [strip_parenthetical(s) for s in raw_segs]
    stripped = [s for s in stripped if s]
    if len(stripped) < 2:
        return None
    if all(word_count(s) <= max_words for s in stripped):
        return stripped
    return None


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"{len(data)} verbetes carregados de {INPUT_PATH}")

    excluded = []
    gloss_stripped = []
    synonym_split = []
    intermediate_records = []
    seen_full_rows = {}
    exact_duplicates = []

    # ---- Etapa 1: exclusão / espécie / remoção de parênteses ------------
    # ---- (agora com is_headword_like em vez do campo type) --------------
    for r in data:
        category = classify(r)

        if category == "hard_exclude":
            excluded.append(r)
            continue

        record = dict(r)

        if category == "gloss_strip":
            before_es, before_pt = record["es"], record["pt"]
            record["es"] = strip_parenthetical(record["es"])
            record["pt"] = strip_parenthetical(record["pt"])
            gloss_stripped.append((r["id"], before_es, record["es"], before_pt, record["pt"]))

        intermediate_records.append((r["id"], category, record))

    # ---- Etapa 2: separa listas de sinônimos em linhas diferentes -------
    # Só o campo 'pt' é separado, porque é o usado no treino hoje
    # (TRAIN_DIRECTIONS é só pt<->tca). O 'es' é limpo (sem parênteses), mas
    # fica num campo só por linha, sem combinar com cada sinônimo pt: ele não é
    # usado no treino e a correspondência entre as listas pt/es não é confiável.
    final_records = []
    for rid, category, record in intermediate_records:
        pt_segs = synonym_segments(record["pt"]) if is_headword_like(record["tca"]) else None

        if record["es"] and is_headword_like(record["tca"]):
            es_segs = synonym_segments(record["es"])
            if es_segs:
                record = dict(record)
                record["es"] = ", ".join(es_segs)  # no mínimo, cada trecho sem as notas entre parênteses

        if pt_segs:
            synonym_split.append((rid, record["pt"], pt_segs))
            for i, seg in enumerate(pt_segs):
                new_record = dict(record)
                new_record["pt"] = seg
                new_record["id"] = f"{rid}_syn{i+1}"
                final_records.append(new_record)
        else:
            final_records.append(record)

    # ---- Etapa 3: duplicados (depois de toda a limpeza, para pegar ------
    # ---- também as linhas que ficaram iguais) ----------------------------
    cleaned_records = []
    for record in final_records:
        key = (record["tca"].strip(), record["es"].strip(), record["pt"].strip())
        if key in seen_full_rows:
            exact_duplicates.append((record["id"], seen_full_rows[key]))
            continue
        seen_full_rows[key] = record["id"]
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

        f.write(f"\n2. PARÊNTESES REMOVIDOS (tradução mantida, nota removida): "
                f"{len(gloss_stripped)}\n" + "-"*60 + "\n")
        for rid, before_es, after_es, before_pt, after_pt in gloss_stripped:
            f.write(f"  id={rid}\n")
            f.write(f"    es: {before_es!r} -> {after_es!r}\n")
            f.write(f"    pt: {before_pt!r} -> {after_pt!r}\n")

        f.write(f"\n3. LISTAS DE SINÔNIMOS SEPARADAS (só o campo pt, usado no treino): "
                f"{len(synonym_split)} verbetes -> {sum(len(s) for _, _, s in synonym_split)} linhas\n"
                + "-"*60 + "\n")
        for rid, original_pt, segs in synonym_split:
            f.write(f"  id={rid}  {original_pt!r} -> {segs}\n")

        f.write(f"\n4. DUPLICADOS REMOVIDOS (mantida a primeira ocorrência): "
                f"{len(exact_duplicates)}\n" + "-"*60 + "\n")
        for dropped_id, kept_id in exact_duplicates:
            f.write(f"  id={dropped_id} duplicado do id={kept_id}\n")

    print(f"\nHard-excluded (dropped): {len(excluded)}")
    print(f"Parênteses removidos: {len(gloss_stripped)}")
    print(f"Listas de sinônimos separadas: {len(synonym_split)} verbetes -> "
          f"{sum(len(s) for _, _, s in synonym_split)} linhas "
          f"(+{sum(len(s) for _, _, s in synonym_split) - len(synonym_split)})")
    print(f"Duplicados removidos: {len(exact_duplicates)}")
    print(f"\nArquivo limpo: {OUTPUT_PATH} ({len(cleaned_records)} verbetes)")
    print(f"Relatório completo: {REPORT_PATH}")


if __name__ == "__main__":
    main()
