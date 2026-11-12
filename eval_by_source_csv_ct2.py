"""
Mesma saída do eval_by_source_full_csv.py (tabela no console + CSV completo),
mas roda o modelo CTranslate2 (int8) em vez do checkpoint PyTorch. Serve para
ver se a quantização do CT2 piorou a tradução, com o mesmo teste e as mesmas
métricas da versão PyTorch.

Monta a entrada com o EOS no final, igual ao translate_cpu.py (sem o EOS a
saída saía sem sentido; esse foi um bug real do projeto).

Não precisa de torch: nem o CT2 nem o tokenizador usam, então as dependências
são as mesmas do translate_cpu.py.

Executar:
    python eval_by_source_csv_ct2.py --model ./nllb_tca_ct2
    python eval_by_source_csv_ct2.py --model ./nllb_tca_ct2 --repetition-penalty 1.3 --no-repeat-ngram-size 3
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import ctranslate2
import sacrebleu
from transformers import AutoTokenizer

import config

def build_lang_code_map():
    """Igual à versão do 03_train.py. Está copiada aqui porque importar aquele
    arquivo executa o código do topo e exige torch, que este script não usa."""
    m = dict(config.NLLB_LANG_CODE)
    m[config.NEW_LANG_KEY] = config.NEW_LANG_CODE
    return m


def has_pathological_repetition(text, n=3, min_repeats=3):
    """True se algum n-grama de palavras aparece min_repeats vezes ou mais no
    texto. É o sinal do loop de repetição (ex.: 'ngẽma chorü pora' repetido
    11+ vezes). Usa n-gramas de palavras porque o tca destes dados separa as
    palavras por espaço."""
    words = text.split()
    if len(words) < n * min_repeats:
        return False
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = {}
    for g in ngrams:
        counts[g] = counts.get(g, 0) + 1
        if counts[g] >= min_repeats:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         help="Pasta do modelo CT2 (gerada pelo optimize_for_deployment.py). "
                              "Precisa ter os arquivos do tokenizador junto, como o "
                              "translate_cpu.py espera.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                         help="O CT2 roda nos dois. Padrão cpu, porque é onde o modelo "
                              "vai ser usado.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1,
                         help="1 = gulosa (rápido). Aumente para uma checagem final mais "
                              "completa depois de escolher o checkpoint.")
    parser.add_argument("--show-examples", type=int, default=5,
                         help="Quantos exemplos (origem / previsão / referência) mostrar "
                              "por grupo. 0 desliga. Não afeta o CSV, que sempre "
                              "tem todos os exemplos.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                         help="1.0 = desligado, igual ao padrão do eval_by_source_full_csv.py, para os "
                              "dois scripts serem comparáveis. Use 1.3 para testar com as "
                              "mesmas proteções do translate_cpu.py.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--csv-output", type=str, default="eval_results_ct2.csv")
    args = parser.parse_args()

    print(f"Carregando modelo CT2: {args.model} (device={args.device})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    translator = ctranslate2.Translator(args.model, device=args.device)

    lang_code_map = build_lang_code_map()

    def lang_token(lang_key):
        return lang_code_map[lang_key]

    # Agrupa os exemplos do teste por (source_file, src_lang, tgt_lang) para ver,
    # por exemplo, "Bíblia pt->tca" separado de "frases tca->pt", igual ao
    # eval_by_source_full_csv.py.
    groups = defaultdict(list)
    with open(config.TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            key = (ex["source_file"], ex["src_lang"], ex["tgt_lang"])
            groups[key].append(ex)

    results = {}
    samples = {}
    csv_rows = []

    for (source_file, src_lang, tgt_lang), examples in groups.items():
        preds, refs, srcs, ids = [], [], [], []
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i:i + args.batch_size]

            # Mesma montagem do Translator.translate() do translate_cpu.py:
            # [token do idioma] + subpalavras + [EOS]. Sem o EOS final a saída saía sem
            # sentido, porque todas as sequências do treino tinham EOS.
            batch_src_tokens = []
            for ex in batch:
                input_ids = tokenizer(ex["src_text"], add_special_tokens=False)["input_ids"]
                tokens = ([lang_token(src_lang)] + tokenizer.convert_ids_to_tokens(input_ids)
                          + [tokenizer.eos_token])
                batch_src_tokens.append(tokens)

            gen_kwargs = dict(
                target_prefix=[[lang_token(tgt_lang)]] * len(batch),
                beam_size=args.num_beams,
                max_decoding_length=config.MAX_LEN,
            )
            if args.repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = args.repetition_penalty
            if args.no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

            results_batch = translator.translate_batch(batch_src_tokens, **gen_kwargs)

            for ex, res in zip(batch, results_batch):
                output_tokens = res.hypotheses[0][1:]  # remove o token de idioma de destino forçado
                output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
                decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
                preds.append(decoded)
                refs.append(ex["tgt_text"])
                srcs.append(ex["src_text"])
                ids.append(ex.get("id", ""))

        chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        rep_rate = 100 * sum(has_pathological_repetition(p) for p in preds) / len(preds)
        len_ratio = sum(len(p.split()) for p in preds) / max(1, sum(len(r.split()) for r in refs))
        results[(source_file, src_lang, tgt_lang)] = (chrf.score, bleu.score, rep_rate, len_ratio, len(examples))
        samples[(source_file, src_lang, tgt_lang)] = list(zip(srcs, preds, refs))[:args.show_examples]

        for ex_id, src, pred, ref in zip(ids, srcs, preds, refs):
            sentence_chrf = sacrebleu.sentence_chrf(pred, [ref], word_order=2).score
            csv_rows.append({
                "source_file": source_file, "src_lang": src_lang, "tgt_lang": tgt_lang,
                "id": ex_id, "source_text": src, "reference_text": ref, "predicted_text": pred,
                "sentence_chrf": round(sentence_chrf, 2),
                "has_repetition": has_pathological_repetition(pred),
                "length_ratio": round(len(pred.split()) / max(1, len(ref.split())), 3),
                "group_chrf": round(chrf.score, 2), "group_bleu": round(bleu.score, 2),
                "group_repeat_pct": round(rep_rate, 1), "group_len_ratio": round(len_ratio, 2),
            })

    print(f"\n{'source_file':<42} {'direction':<10} {'n':>5} {'chrF++':>8} {'BLEU':>7} {'repeat%':>8} {'len ratio':>10}")
    print("-" * 100)
    for (source_file, src_lang, tgt_lang), (chrf_s, bleu_s, rep, lr, n) in sorted(results.items()):
        print(f"{source_file:<42} {src_lang}->{tgt_lang:<7} {n:>5} {chrf_s:>8.2f} {bleu_s:>7.2f} {rep:>7.1f}% {lr:>10.2f}")

    print(f"\n{'source_file':<42} {'n':>5} {'chrF++':>8} {'BLEU':>7} {'repeat%':>8} {'len ratio':>10}")
    print("-" * 90)
    by_file = defaultdict(list)
    for (source_file, _, _), (chrf_s, bleu_s, rep, lr, n) in results.items():
        by_file[source_file].append((chrf_s, bleu_s, rep, lr, n))
    for source_file, vals in sorted(by_file.items()):
        total_n = sum(n for _, _, _, _, n in vals)
        w_chrf = sum(c * n for c, _, _, _, n in vals) / total_n if total_n else 0
        w_bleu = sum(b * n for _, b, _, _, n in vals) / total_n if total_n else 0
        w_rep = sum(r * n for _, _, r, _, n in vals) / total_n if total_n else 0
        w_lr = sum(l * n for _, _, _, l, n in vals) / total_n if total_n else 0
        print(f"{source_file:<42} {total_n:>5} {w_chrf:>8.2f} {w_bleu:>7.2f} {w_rep:>7.1f}% {w_lr:>10.2f}")

    if args.show_examples > 0:
        print(f"\n{'='*70}\nExemplos de tradução ({args.show_examples} por grupo)\n{'='*70}")
        for (source_file, src_lang, tgt_lang) in sorted(samples.keys()):
            chrf_s, bleu_s, rep, lr, n = results[(source_file, src_lang, tgt_lang)]
            print(f"\n--- {source_file}  {src_lang}->{tgt_lang}  "
                  f"(chrF++={chrf_s:.2f}, BLEU={bleu_s:.2f}, repeat%={rep:.1f}, n={n}) ---")
            for src, pred, ref in samples[(source_file, src_lang, tgt_lang)]:
                print(f"  src:  {src}")
                print(f"  pred: {pred}")
                print(f"  ref:  {ref}")
                print()

    fieldnames = ["source_file", "src_lang", "tgt_lang", "id", "source_text", "reference_text",
                  "predicted_text", "sentence_chrf", "has_repetition", "length_ratio",
                  "group_chrf", "group_bleu", "group_repeat_pct", "group_len_ratio"]
    with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n{len(csv_rows)} linhas gravadas em {args.csv_output}")


if __name__ == "__main__":
    main()