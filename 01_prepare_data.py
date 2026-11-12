"""
Passo 1: preparação dos dados.

Lê os arquivos JSON de origem, monta os pares (origem, destino) para cada
direção em config.TRAIN_DIRECTIONS e grava train.jsonl e test.jsonl.

Como o teste é separado:
    - O teste vem de duas fontes, cada uma com sua cota:
        1. Bíblia (antigotestamento_pt_tca e novotestamento_pt_tca): a maior
           parte, TEST_SIZE_TOTAL_EXAMPLES - TEST_PHRASES_COUNT exemplos.
        2. Frases do dicionário (dicionario_es_pt_tca_phrases): cota fixa de
           TEST_PHRASES_COUNT, para o teste não ficar só no registro bíblico
           sem gastar muito dos poucos dados não bíblicos.
    - Por padrão são 1000 exemplos: 850 da Bíblia e 150 do dicionário.
    - A separação é feita pelo id do registro, não pela direção. Se uma
      frase cai no teste, nenhuma das direções dela (pt->tca, tca->pt) vai
      para o treino. Assim o modelo não vê a mesma frase no treino.
    - O resto (Bíblia, frases e pares de palavras do dicionário, e todo o
      livrosprogramacao) vai para o treino.

Executar:
    python 01_prepare_data.py
"""

import json
import random
from collections import defaultdict

import config
from text_utils import normalize_text


def load_records(path, langs):
    """Lê o arquivo JSON e devolve (id, {idioma: texto}) para os registros
    que têm pelo menos dois idiomas preenchidos."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"{path}: esperava uma lista de registros em JSON, veio {type(data)}. "
            f"Ajuste load_records() se o arquivo tiver outro formato."
        )

    for rec in data:
        rid = rec.get("id")
        texts = {}
        for lang in langs:
            val = rec.get(lang)
            if isinstance(val, str) and val.strip():
                texts[lang] = normalize_text(val)
        if len(texts) >= 2:
            yield rid, texts


def expand_directions(texts, directions):
    """Recebe {'pt': ..., 'tca': ..., 'es': ...} e gera os pares de todas as
    direções pedidas que existem nesse registro."""
    pairs = []
    for src, tgt in directions:
        if src in texts and tgt in texts:
            pairs.append((src, tgt, texts[src], texts[tgt]))
    return pairs


def main():
    random.seed(config.RANDOM_SEED)
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)

    # grupo -> lista de (record_id, arquivo_origem, textos)
    records_by_file = defaultdict(list)
    for fname, meta in config.FILES.items():
        path = config.DATA_DIR / fname
        count = 0
        for rid, texts in load_records(path, meta["langs"]):
            records_by_file[fname].append((rid, texts))
            count += 1
        print(f"{count:>6} registros válidos carregados de {fname}")

    def build_test_pool(eligible_files, budget, label):
        """Embaralha os registros de eligible_files e vai separando registros
        inteiros (pelo id, não pela direção) até chegar na cota. Assim a mesma
        frase não aparece no treino numa direção e no teste na outra.
        Retorna (test_records, held_out_keys, count)."""
        pool = []
        for fname in eligible_files:
            for rid, texts in records_by_file.get(fname, []):
                pool.append((fname, rid, texts))
        random.shuffle(pool)

        picked = []
        held = set()
        count = 0
        for fname, rid, texts in pool:
            if count >= budget:
                break
            pairs = expand_directions(texts, config.TRAIN_DIRECTIONS)
            if not pairs:
                continue
            if count + len(pairs) > budget:
                continue
            picked.append((fname, rid, texts, pairs))
            held.add((fname, rid))
            count += len(pairs)

        print(f"{label}: {len(picked)} registros -> {count} exemplos (cota: {budget})")
        return picked, held, count

    # ---- Teste da Bíblia (principal) --------------------------------------
    bible_budget = config.TEST_SIZE_TOTAL_EXAMPLES - config.TEST_PHRASES_COUNT
    bible_test_records, bible_held, bible_count = build_test_pool(
        config.TEST_ELIGIBLE_FILES, bible_budget, "Teste Bíblia"
    )

    # ---- Teste das frases do dicionário (cota fixa) ------------------------
    phrases_test_records, phrases_held, phrases_count = build_test_pool(
        {config.TEST_PHRASES_FILE}, config.TEST_PHRASES_COUNT, "Teste frases"
    )

    test_records = bible_test_records + phrases_test_records
    held_out_keys = bible_held | phrases_held
    test_examples_count = bible_count + phrases_count

    print(f"\nTeste total: {len(test_records)} registros -> {test_examples_count} exemplos "
          f"(cota: {config.TEST_SIZE_TOTAL_EXAMPLES} = "
          f"{bible_budget} Bíblia + {config.TEST_PHRASES_COUNT} frases do dicionário)")

    # ---- Treino: tudo que não foi para o teste ----------------------------
    train_examples = []
    tca_sentences = set()
    raw_counts_by_file = defaultdict(int)      # antes do oversampling
    example_counts_by_file = defaultdict(int)  # depois do oversampling
    example_counts_by_group = defaultdict(int)

    for fname, meta in config.FILES.items():
        group = meta.get("group", "other")
        factor = config.OVERSAMPLE_FACTOR.get(group, 1)

        for rid, texts in records_by_file.get(fname, []):
            if (fname, rid) in held_out_keys:
                continue
            pairs = expand_directions(texts, config.TRAIN_DIRECTIONS)
            raw_counts_by_file[fname] += len(pairs)

            # Oversampling: repete os pares deste registro `factor` vezes. Só afeta
            # o treino; o teste já foi separado acima.
            for _ in range(factor):
                for src, tgt, src_text, tgt_text in pairs:
                    train_examples.append({
                        "id": rid,
                        "source_file": fname,
                        "src_lang": src,
                        "tgt_lang": tgt,
                        "src_text": src_text,
                        "tgt_text": tgt_text,
                    })
                    example_counts_by_file[fname] += 1
                    example_counts_by_group[group] += 1

            if config.NEW_LANG_KEY in texts:
                tca_sentences.add(texts[config.NEW_LANG_KEY])

    print("\nExemplos por arquivo (antes -> depois do oversampling):")
    for fname in config.FILES:
        group = config.FILES[fname].get("group", "other")
        factor = config.OVERSAMPLE_FACTOR.get(group, 1)
        print(f"  {fname:<32} grupo={group:<6} fator={factor}x  "
              f"{raw_counts_by_file[fname]:>6} -> {example_counts_by_file[fname]:>6}")

    total_examples = sum(example_counts_by_group.values())
    print("\nExemplos de treino por grupo:")
    for group, count in example_counts_by_group.items():
        pct = 100 * count / total_examples if total_examples else 0
        print(f"  {group:<8} {count:>6} exemplos ({pct:.1f}%)")

    test_examples = []
    for fname, rid, texts, pairs in test_records:
        for src, tgt, src_text, tgt_text in pairs:
            test_examples.append({
                "id": rid,
                "source_file": fname,
                "src_lang": src,
                "tgt_lang": tgt,
                "src_text": src_text,
                "tgt_text": tgt_text,
            })

    random.shuffle(train_examples)

    with open(config.TRAIN_JSONL, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(config.TEST_JSONL, "w", encoding="utf-8") as f:
        for ex in test_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Texto monolíngue em Ticuna para estender o tokenizador no passo 2.
    # Só usa o treino, para o teste não ser visto antes da avaliação.
    with open(config.TCA_MONOLINGUAL_TXT, "w", encoding="utf-8") as f:
        for s in sorted(tca_sentences):
            f.write(s + "\n")

    print(f"\n{len(train_examples)} exemplos de treino gravados em {config.TRAIN_JSONL}")
    print(f"{len(test_examples)} exemplos de teste gravados em {config.TEST_JSONL}")
    print(f"{len(tca_sentences)} frases em Ticuna gravadas em {config.TCA_MONOLINGUAL_TXT}")

    # confere se nenhum id do teste foi parar no treino
    train_ids_by_file = defaultdict(set)
    for ex in train_examples:
        train_ids_by_file[ex["source_file"]].add(ex["id"])
    leaks = [k for k in held_out_keys if k[1] in train_ids_by_file.get(k[0], set())]
    assert not leaks, f"Vazamento encontrado: {leaks[:5]}"
    print("\nSem vazamento: nenhum registro do teste aparece no treino.")


if __name__ == "__main__":
    main()
