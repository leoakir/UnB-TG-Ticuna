"""
Treina um detector simples de idioma (n-gramas de caracteres) para separar
pt de tca, usando os dados reais de treino. Ferramentas prontas de
identificação de idioma não servem aqui porque nunca viram Ticuna. Salva um
perfil JSON pequeno que o translate_cpu.py usa para detectar a direção da
tradução automaticamente.

Rodar uma vez na máquina de treino, depois do treino final:
    python build_lang_detector.py

Depois copie o lang_detector.json para a máquina de uso junto com o modelo
CT2 (o optimize_for_deployment.py já faz isso se rodar depois deste script).
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import config


def char_ngrams(text, n_sizes=(2, 3)):
    text = text.lower()
    grams = []
    for n in n_sizes:
        grams.extend(text[i:i+n] for i in range(len(text) - n + 1))
    return grams


def build_profile(sentences):
    counts = defaultdict(int)
    for s in sentences:
        for g in char_ngrams(s):
            counts[g] += 1
    return dict(counts)


def main():
    pt_sentences = []
    tca_sentences = []

    # Pega as frases do train.jsonl. Cada registro tem src_lang/src_text e
    # tgt_lang/tgt_text; guarda o lado que for pt ou tca.
    for jsonl_path in (config.TRAIN_JSONL, config.TEST_JSONL):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                if ex["src_lang"] == "pt":
                    pt_sentences.append(ex["src_text"])
                elif ex["src_lang"] == "tca":
                    tca_sentences.append(ex["src_text"])

    print(f"{len(pt_sentences)} frases pt e {len(tca_sentences)} frases tca coletadas.")

    pt_counts = build_profile(pt_sentences)
    tca_counts = build_profile(tca_sentences)
    pt_total = sum(pt_counts.values())
    tca_total = sum(tca_counts.values())
    vocab_size = len(set(pt_counts) | set(tca_counts))

    profile = {
        "pt_counts": pt_counts,
        "tca_counts": tca_counts,
        "pt_total": pt_total,
        "tca_total": tca_total,
        "vocab_size": vocab_size,
    }

    out_path = Path("lang_detector.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False)
    print(f"Perfil do detector salvo em {out_path} "
          f"({out_path.stat().st_size / 1024:.0f} KB, {vocab_size} n-gramas distintos)")

    # Teste rápido numa amostra aleatória dos próprios dados, para ter uma
    # acurácia de verdade antes de usar o detector.
    import random
    random.seed(0)
    test_pt = random.sample(pt_sentences, min(200, len(pt_sentences)))
    test_tca = random.sample(tca_sentences, min(200, len(tca_sentences)))

    def score(text, counts, total):
        log_prob = 0.0
        for g in char_ngrams(text):
            log_prob += math.log((counts.get(g, 0) + 1) / (total + vocab_size))
        return log_prob

    correct = 0
    for s in test_pt:
        correct += score(s, pt_counts, pt_total) > score(s, tca_counts, tca_total)
    for s in test_tca:
        correct += score(s, tca_counts, tca_total) > score(s, pt_counts, pt_total)
    total_n = len(test_pt) + len(test_tca)
    print(f"\nAcurácia em {total_n} frases da amostra: {correct}/{total_n} "
          f"({100*correct/total_n:.1f}%)")


if __name__ == "__main__":
    main()
