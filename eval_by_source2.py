"""
Avalia um checkpoint no test.jsonl e mostra o chrF++ separado por arquivo de
origem (e por direção), em vez de uma média única. Uma média só esconderia se
o modelo vai bem no texto bíblico e mal nas frases mais gerais, ou o
contrário. É para isso que existe a parte do teste com frases do dicionário.

Igual ao eval_by_source.py, com opções para combater repetição na geração.

Executar:
    python eval_by_source2.py --checkpoint ./nllb_tca_checkpoints/checkpoint-600
    python eval_by_source2.py   # usa o checkpoint mais recente
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config

# Reaproveita TranslationDataset/build_lang_code_map do 03_train.py sem
# import normal (o nome do arquivo começa com número).
import importlib.util
spec = importlib.util.spec_from_file_location("train_mod", "03_train.py")
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)


def find_latest_checkpoint():
    checkpoints = sorted(
        config.MODEL_OUTPUT_DIR.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        raise SystemExit(f"Nenhum checkpoint encontrado em {config.MODEL_OUTPUT_DIR}")
    return checkpoints[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=1,
                         help="1 = gulosa (rápido). Aumente para uma checagem final mais "
                              "completa depois de escolher o checkpoint.")
    parser.add_argument("--show-examples", type=int, default=5,
                         help="Quantos exemplos (origem / previsão / referência) mostrar "
                              "por grupo, para ler as traduções e não só a nota. "
                              "0 desliga.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                         help="Penaliza tokens repetidos na geração "
                              "(1.0 = sem penalidade, padrão; teste 1.3-1.5 para ver "
                              "se os loops de repetição em frases longas se resolvem "
                              "na decodificação, sem precisar treinar mais).")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0,
                         help="Impede o modelo de repetir qualquer n-grama "
                              "desse tamanho (0 = desligado; use 3 para bloquear "
                              "loops como 'ngẽma chorü pora' se repetindo).")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    print(f"Carregando checkpoint: {ckpt_dir}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir).to("cuda")
    model.eval()

    lang_code_map = train_mod.build_lang_code_map()

    # Agrupa os exemplos do teste por (source_file, src_lang, tgt_lang) para ver,
    # por exemplo, "Bíblia pt->tca" separado de "frases tca->pt".
    groups = defaultdict(list)
    with open(config.TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            key = (ex["source_file"], ex["src_lang"], ex["tgt_lang"])
            groups[key].append(ex)

    def lang_id(lang_key):
        return tokenizer.convert_tokens_to_ids(lang_code_map[lang_key])

    results = {}
    samples = {}
    for (source_file, src_lang, tgt_lang), examples in groups.items():
        preds, refs, srcs = [], [], []
        for i in range(0, len(examples), args.batch_size):
            batch = examples[i:i + args.batch_size]
            src_texts = [ex["src_text"] for ex in batch]
            ref_texts = [ex["tgt_text"] for ex in batch]

            enc = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True,
                             max_length=config.MAX_LEN)
            # Coloca o token do idioma de origem no início, igual ao treino
            # (ver TranslationDataset no 03_train.py).
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

        chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
        results[(source_file, src_lang, tgt_lang)] = (chrf.score, len(examples))
        samples[(source_file, src_lang, tgt_lang)] = list(zip(srcs, preds, refs))[:args.show_examples]

    print(f"\n{'source_file':<42} {'direction':<10} {'n':>5} {'chrF++':>8}")
    print("-" * 70)
    for (source_file, src_lang, tgt_lang), (score, n) in sorted(results.items()):
        print(f"{source_file:<42} {src_lang}->{tgt_lang:<7} {n:>5} {score:>8.2f}")

    # Mostra também o total por arquivo de origem (todas as direções juntas)
    print(f"\n{'source_file':<42} {'n':>5} {'chrF++':>8}")
    print("-" * 60)
    by_file = defaultdict(list)
    for (source_file, _, _), (score, n) in results.items():
        by_file[source_file].append((score, n))
    for source_file, scores in sorted(by_file.items()):
        total_n = sum(n for _, n in scores)
        weighted = sum(s * n for s, n in scores) / total_n if total_n else 0
        print(f"{source_file:<42} {total_n:>5} {weighted:>8.2f}")

    if args.show_examples > 0:
        print(f"\n{'='*70}\nExemplos de tradução ({args.show_examples} por grupo)\n{'='*70}")
        for (source_file, src_lang, tgt_lang) in sorted(samples.keys()):
            score, n = results[(source_file, src_lang, tgt_lang)]
            print(f"\n--- {source_file}  {src_lang}->{tgt_lang}  (chrF++={score:.2f}, n={n}) ---")
            for src, pred, ref in samples[(source_file, src_lang, tgt_lang)]:
                print(f"  src:  {src}")
                print(f"  pred: {pred}")
                print(f"  ref:  {ref}")
                print()


if __name__ == "__main__":
    main()
