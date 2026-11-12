"""
Limpeza do dicionario_MADLAD_10b.json.

Separa o dicionário em DOIS arquivos:
  - dicionario_MADLAD_10b_word_pairs.json: verbetes curtos de vocabulário
    (o tca é uma palavra ou expressão curta, não uma frase)
  - dicionario_MADLAD_10b_phrases.json: frases de exemplo completas

Nos pares de palavras:
  1. EXCLUI verbetes que só explicam a gramática, sem dar uma palavra
     (ex.: "-e" -> "Se añade al verbo para indicar plural."). Não são
     tradução e ensinariam errado o modelo.
  2. Separa listas de sinônimos com vírgula em pt e em es (ex.: "aço,
     ferro") e junta pela posição: 1º sinônimo pt com 1º es, 2º com 2º etc.
     Se as quantidades forem diferentes, os sinônimos que sobram no fim da
     lista maior são descartados, para cada linha ter um pt e um es.
  3. Notas entre parênteses (ex.: "bijao (hojas grandes...)") são removidas
     na mesma etapa. Um campo sem vírgula é tratado como um trecho só, então
     o mesmo código cuida dos parênteses e dos sinônimos.
  4. Remove linhas duplicadas, inclusive as que ficaram iguais depois da
     limpeza (ex.: "mergulhar, mergulhar (profundamente)" vira "mergulhar"
     duas vezes).

Executar:
    python clean_dictionary.py
"""

import json
import re
from pathlib import Path

INPUT_PATH = Path("./raw_data/dicionario_MADLAD_10b.json")
WORD_PAIRS_OUTPUT = Path("./raw_data/dicionario_MADLAD_10b_word_pairs.json")
PHRASES_OUTPUT = Path("./raw_data/dicionario_MADLAD_10b_phrases.json")
REPORT_PATH = Path("./work/dictionary_cleaning_report.txt")

EXPLANATION_MARKERS = [
    "se usa", "se refiere", "indica", "se emplea", "denota",
    "sirve para", "se dice", "se utiliza", "expresa", "se añade",
]

SYNONYM_SPLIT_MAX_WORDS = 4


def word_count(s):
    return len(s.split())


def is_headword_like(tca_text):
    """Detecta verbetes/pares de palavras pelo conteúdo (o campo 'type' não é
    confiável: ~1.305 verbetes reais têm esse campo vazio)."""
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
    """Retorna 'hard_exclude' ou 'keep'. Só é chamada para verbetes curtos
    (candidatos a par de palavras); as frases passam direto."""
    if has_explanation_marker(record["es"]):
        has_parens = "(" in record["es"] and ")" in record["es"]
        if has_parens:
            return "keep"  # palavra com nota, ex.: "aura -> añuje caspi (se usa para canoas)"
        if is_species_id(record["es"]):
            return "keep"  # não há tradução melhor para esta espécie
        return "hard_exclude"  # só nota gramatical, sem nenhuma palavra
    return "keep"


def field_segments(text, max_words=SYNONYM_SPLIT_MAX_WORDS):
    """Trechos limpos de um campo. Uma lista curta de sinônimos (cada trecho
    entre vírgulas com até max_words palavras, sem os parênteses) vira vários
    trechos. Qualquer outra coisa (um valor só, ou uma frase longa que tem
    vírgula) vira UM trecho: o campo inteiro sem os parênteses. A mesma
    função cuida dos sinônimos e dos parênteses."""
    text = text.strip()
    if not text:
        return []
    raw_segs = [s.strip() for s in text.split(",") if s.strip()]
    if len(raw_segs) < 2:
        return [strip_parenthetical(text)]
    stripped = [strip_parenthetical(s) for s in raw_segs]
    stripped = [s for s in stripped if s]
    if len(stripped) >= 2 and all(word_count(s) <= max_words for s in stripped):
        return stripped
    return [strip_parenthetical(text)]


def dedupe(records, id_key="id"):
    seen = {}
    out = []
    dupes = []
    for r in records:
        key = (r["tca"].strip(), r["es"].strip(), r["pt"].strip())
        if key in seen:
            dupes.append((r[id_key], seen[key]))
            continue
        seen[key] = r[id_key]
        out.append(r)
    return out, dupes


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"{len(data)} verbetes carregados de {INPUT_PATH}")

    phrases_raw = [r for r in data if not is_headword_like(r["tca"])]
    word_candidates = [r for r in data if is_headword_like(r["tca"])]
    print(f"Separação pelo conteúdo: {len(word_candidates)} candidatos a par de palavras, "
          f"{len(phrases_raw)} frases")

    # ---- Pares de palavras: exclusão e depois separação dos sinônimos ----
    excluded = []
    synonym_split = []
    word_pairs_raw = []

    for r in word_candidates:
        if classify(r) == "hard_exclude":
            excluded.append(r)
            continue

        pt_segs = field_segments(r["pt"])
        es_segs = field_segments(r["es"])
        common_len = min(len(pt_segs), len(es_segs))
        if common_len == 0:
            continue  # não deveria acontecer (os campos nunca estão vazios), só por segurança

        discarded_pt = pt_segs[common_len:]
        discarded_es = es_segs[common_len:]
        if discarded_pt or discarded_es:
            synonym_split.append((r["id"], r["pt"], r["es"], pt_segs[:common_len],
                                   es_segs[:common_len], discarded_pt, discarded_es))

        for i in range(common_len):
            new_record = dict(r)
            new_record["pt"] = pt_segs[i]
            new_record["es"] = es_segs[i]
            if common_len > 1:
                new_record["id"] = f"{r['id']}_syn{i+1}"
            word_pairs_raw.append(new_record)

    word_pairs, word_pair_dupes = dedupe(word_pairs_raw)
    phrases, phrase_dupes = dedupe(phrases_raw)

    Path("./raw_data").mkdir(parents=True, exist_ok=True)
    Path("./work").mkdir(parents=True, exist_ok=True)

    with open(WORD_PAIRS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(word_pairs, f, ensure_ascii=False, indent=2)
    with open(PHRASES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(phrases, f, ensure_ascii=False, indent=2)

    # ---- Relatório ---------------------------------------------------------
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Relatório de limpeza do dicionário\n{'='*60}\n\n")
        f.write(f"Entrada: {len(data)} verbetes\n")
        f.write(f"  -> {len(word_candidates)} candidatos a par de palavras, {len(phrases_raw)} frases\n\n")
        f.write(f"Saída: {WORD_PAIRS_OUTPUT.name} ({len(word_pairs)} linhas), "
                f"{PHRASES_OUTPUT.name} ({len(phrases)} linhas)\n\n")

        f.write(f"1. EXCLUÍDOS (só explicação gramatical, sem palavra): "
                f"{len(excluded)}\n" + "-"*60 + "\n")
        for r in excluded:
            f.write(f"  id={r['id']:>5}  tca={r['tca']!r:20}  es={r['es']!r}\n")

        f.write(f"\n2. PAREAMENTO DE SINÔNIMOS (pt/es separados e juntados pela posição): "
                f"{len(synonym_split)} verbetes precisaram de corte por quantidades diferentes\n"
                + "-"*60 + "\n")
        for rid, orig_pt, orig_es, kept_pt, kept_es, disc_pt, disc_es in synonym_split:
            f.write(f"  id={rid}\n")
            f.write(f"    original pt={orig_pt!r}  es={orig_es!r}\n")
            f.write(f"    pares mantidos: {list(zip(kept_pt, kept_es))}\n")
            if disc_pt:
                f.write(f"    pt descartados (sobraram, sem es): {disc_pt}\n")
            if disc_es:
                f.write(f"    es descartados (sobraram, sem pt): {disc_es}\n")

        f.write(f"\n3. DUPLICADOS REMOVIDOS: word_pairs: {len(word_pair_dupes)}, "
                f"phrases: {len(phrase_dupes)}\n" + "-"*60 + "\n")
        for dropped_id, kept_id in word_pair_dupes:
            f.write(f"  [word_pairs] id={dropped_id} duplicado do id={kept_id}\n")
        for dropped_id, kept_id in phrase_dupes:
            f.write(f"  [phrases] id={dropped_id} duplicado do id={kept_id}\n")

    print(f"\nHard-excluded: {len(excluded)}")
    print(f"Verbetes com corte por quantidade diferente de pt/es: {len(synonym_split)}")
    print(f"Duplicados removidos: {len(word_pair_dupes)} (word_pairs), {len(phrase_dupes)} (phrases)")
    print(f"\n{WORD_PAIRS_OUTPUT}: {len(word_pairs)} linhas")
    print(f"{PHRASES_OUTPUT}: {len(phrases)} linhas")
    print(f"Relatório completo: {REPORT_PATH}")


if __name__ == "__main__":
    main()
