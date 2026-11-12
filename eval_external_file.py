"""
Mesma saída do eval_by_source_full_csv.py (tabela no console + CSV completo),
mas lê um arquivo externo com os campos tca/pt/(es) direto em cada registro,
como o ticuna_salud_parallel_pt_clean.json, em vez do train.jsonl/test.jsonl
do pipeline. Serve para qualquer arquivo nesse formato: basta passar em
--test-file.

Atenção (o aviso também aparece na saída): se o campo pt tiver
pt_source == "mt_generated" (caso dos 194 registros deste arquivo), a
referência é uma tradução automática, provavelmente do espanhol, e não uma
tradução humana. A nota mede a concordância com o português de outro sistema
de tradução, não com a resposta certa. É um sinal mais fraco que as
avaliações com referência humana (Bíblia e dicionário), mas ainda serve para
ver como o modelo generaliza para um domínio que não está no treino.

Executar:
    python eval_external_file.py --test-file ticuna_salud_parallel_pt_clean.json
    python eval_external_file.py --test-file outro_arquivo.json --checkpoint ./nllb_tca_checkpoints/final
"""

import argparse
import csv
import json
from collections import defaultdict, Counter
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config
from text_utils import normalize_text

# Reaproveita TranslationDataset/build_lang_code_map do 03_train.py sem
# import normal (o nome do arquivo começa com número).
import importlib.util
spec = importlib.util.spec_from_file_location("train_mod", "03_train.py")
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)


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


def find_latest_checkpoint():
    checkpoints = sorted(
        config.MODEL_OUTPUT_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    final = config.MODEL_OUTPUT_DIR / "final"
    if final.exists():
        return final
    if not checkpoints:
        raise SystemExit(f"Nenhum checkpoint encontrado em {config.MODEL_OUTPUT_DIR}")
    return checkpoints[-1]


def load_external_records(path):
    """Lê um arquivo JSONL com tca/pt/(es) em cada registro e gera os pares
    pt<->tca (como no TRAIN_DIRECTIONS do projeto; o es é carregado se existir,
    mas não é usado). Normaliza o texto igual ao treino e agrupa pelo campo
    'source' do registro (ou pelo nome do arquivo, se não houver)."""
    groups = defaultdict(list)
    pt_source_counts = Counter()

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            source_file = rec.get("source", Path(path).name)
            rec_id = f"{source_file}_{rec.get('page', line_num)}"
            pt_source_counts[rec.get("pt_source", "unknown")] += 1

            tca_text = normalize_text(rec.get("tca", "").strip())
            pt_text = normalize_text(rec.get("pt", "").strip())
            if not tca_text or not pt_text:
                continue  # pula registros incompletos em vez de dar erro

            for src_lang, tgt_lang, src_text, tgt_text in [
                ("pt", "tca", pt_text, tca_text),
                ("tca", "pt", tca_text, pt_text),
            ]:
                key = (source_file, src_lang, tgt_lang)
                groups[key].append({
                    "id": rec_id, "src_text": src_text, "tgt_text": tgt_text,
                })

    return groups, pt_source_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", type=str, required=True,
                         help="Caminho do arquivo JSONL externo com tca/pt/(es).")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1,
                         help="1 = gulosa (rápido). Aumente para uma checagem final mais "
                              "completa depois de escolher o checkpoint.")
    parser.add_argument("--show-examples", type=int, default=5,
                         help="Quantos exemplos (origem / previsão / referência) mostrar "
                              "por grupo. 0 desliga. Não afeta o CSV, que sempre "
                              "tem todos os exemplos.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--csv-output", type=str, default="eval_external_results.csv")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    print(f"Carregando checkpoint: {ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).to("cuda")
    model.eval()

    lang_code_map = train_mod.build_lang_code_map()

    groups, pt_source_counts = load_external_records(args.test_file)
    total_records = sum(len(v) for v in groups.values()) // 2  # 2 direções por registro
    print(f"{total_records} registros carregados de {args.test_file} "
          f"-> {sum(len(v) for v in groups.values())} exemplos (contando as direções)")

    non_human = {k: v for k, v in pt_source_counts.items() if k != "human"}
    if non_human:
        print(f"\n{'!'*70}")
        print(f"ATENÇÃO: origem do pt = {dict(pt_source_counts)}")
        print(f"Nos registros sem pt_source=='human' a referência em pt é uma tradução "
              f"automática, não humana. A nota mede a concordância com outro sistema "
              f"de tradução, não com a resposta certa;")
        print(f"é um sinal mais fraco que as avaliações com referência humana do projeto.")
        print(f"{'!'*70}\n")

    def lang_id(lang_key):
        return tokenizer.convert_tokens_to_ids(lang_code_map[lang_key])

    results = {}
    samples = {}
    csv_rows = []

    for (source_file, src_lang, tgt_lang), examples in groups.items():
        preds, refs, srcs, ids = [], [], [], []
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i:i + args.batch_size]
            src_texts = [ex["src_text"] for ex in batch]
            ref_texts = [ex["tgt_text"] for ex in batch]
            batch_ids = [ex["id"] for ex in batch]

            enc = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True,
                             max_length=config.MAX_LEN)
            src_id = lang_id(src_lang)
            input_ids = torch.cat([
                torch.full((enc["input_ids"].shape[0], 1), src_id, dtype=torch.long),
                enc["input_ids"],
            ], dim=1).to("cuda")
            attention_mask = torch.cat([
                torch.ones((enc["attention_mask"].shape[0], 1), dtype=torch.long),
                enc["attention_mask"],
            ], dim=1).to("cuda")

            gen_kwargs = dict(
                input_ids=input_ids, attention_mask=attention_mask,
                forced_bos_token_id=lang_id(tgt_lang),
                max_length=config.MAX_LEN, num_beams=args.num_beams,
            )
            if args.repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = args.repetition_penalty
            if args.no_repeat_ngram_size > 0:
                gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

            with torch.no_grad():
                generated = model.generate(**gen_kwargs)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            preds.extend(decoded)
            refs.extend(ref_texts)
            srcs.extend(src_texts)
            ids.extend(batch_ids)

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
