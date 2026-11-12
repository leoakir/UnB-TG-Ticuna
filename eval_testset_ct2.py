"""
eval_testset_ct2.py

Avalia um teste paralelo por<->tca em TSV (ex.: ticuna_pt_testset.tsv, com as
colunas id, section, paragraph, type, por, tca) usando o modelo NLLB convertido
para CTranslate2 (int8) na CPU. Grava um CSV por exemplo no mesmo formato dos
outros scripts de avaliação (eval_by_source_full_csv.py /
eval_by_source_csv_ct2.py / build_review_sample.py), então o arquivo pode ir
direto para o build_review_sample.py como --standard-csv ou --ood-csv.

Monta a entrada do mesmo jeito que o resto do projeto
(TranslationDataset._encode() do 03_train.py e translate_cpu.py):
    tokens de entrada = [token do idioma] + subpalavras + [EOS]
    a geração começa com o token do idioma de destino via target_prefix,
    como na receita do CTranslate2 para o NLLB.

Não depende de torch nem do config.py do projeto (os dois supõem uma máquina
específica). Só precisa de ctranslate2, transformers (só o AutoTokenizer) e
sacrebleu.

Dependências:
    pip install ctranslate2 transformers sacrebleu sentencepiece

Uso:
    python eval_testset_ct2.py \
        --model-dir nllb_tca_ct2 \
        --testset-tsv ticuna_pt_testset.tsv \
        --output-csv eval_testset_results.csv \
        --directions both

    # Testa a lógica do script (leitura do TSV, chrF, detecção de repetição,
    # formato do CSV) sem modelo e sem traduções reais:
    python eval_testset_ct2.py --dry-run \
        --testset-tsv ticuna_pt_testset.tsv \
        --output-csv eval_testset_results_dryrun.csv
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import sacrebleu
except ImportError:
    raise SystemExit("Falta uma dependência: pip install sacrebleu")


DEFAULT_LANG_CODES = {
    "por": "por_Latn",
    "pt": "por_Latn",
    "tca": "tca_Latn",
}

# Detector de repetição de n-gramas, com o mesmo limite dos outros scripts
# de avaliação (n-grama repetido 3 vezes ou mais indica loop de repetição).
def has_pathological_repetition(text, n=3, min_repeats=3):
    tokens = text.split()
    if len(tokens) < n * min_repeats:
        return False
    ngram_counts = defaultdict(int)
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngram_counts[ngram] += 1
        if ngram_counts[ngram] >= min_repeats:
            return True
    return False


def read_testset_tsv(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"TSV do teste não encontrado: {path}")
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"id", "section", "paragraph", "type", "por", "tca"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"Falta(m) coluna(s) obrigatória(s) no TSV: {sorted(missing)}. "
                f"Colunas encontradas: {reader.fieldnames}"
            )
        for r in reader:
            por = (r.get("por") or "").strip()
            tca = (r.get("tca") or "").strip()
            if not por or not tca:
                continue  # pula linhas incompletas (ex.: linha em branco no final)
            rows.append(r)
    if not rows:
        raise SystemExit(f"Nenhuma linha válida (por, tca) em {path}")
    return rows


class Ct2Translator:
    """Monta a entrada do CT2 igual ao translate_cpu.py: [token do idioma] +
    subpalavras + [EOS], target_prefix para forçar o idioma de destino, e
    decodifica com skip_special_tokens.
    """

    def __init__(self, model_dir, lang_codes=None, beam_size=4,
                 no_repeat_ngram_size=3, repetition_penalty=1.3, max_len=200):
        import ctranslate2
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        ct2_dir = model_dir / "ct2" if (model_dir / "ct2").exists() else model_dir
        tok_dir = model_dir if (model_dir / "tokenizer_config.json").exists() else ct2_dir

        self.translator = ctranslate2.Translator(str(ct2_dir), device="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
        self.lang_codes = lang_codes or DEFAULT_LANG_CODES
        self.beam_size = beam_size
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty
        self.max_len = max_len

    def translate(self, text, src_lang, tgt_lang):
        src_code = self.lang_codes[src_lang]
        tgt_code = self.lang_codes[tgt_lang]

        input_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        src_tokens = (
            [src_code]
            + self.tokenizer.convert_ids_to_tokens(input_ids)
            + [self.tokenizer.eos_token]
        )

        result = self.translator.translate_batch(
            [src_tokens],
            target_prefix=[[tgt_code]],
            beam_size=self.beam_size,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            repetition_penalty=self.repetition_penalty,
            max_decoding_length=self.max_len,
        )
        out_tokens = result[0].hypotheses[0][1:]  # remove o token de idioma de destino forçado
        out_ids = self.tokenizer.convert_tokens_to_ids(out_tokens)
        return self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()


class DryRunTranslator:
    """Usado no --dry-run: devolve a referência com uma pequena alteração fixa,
    só para testar o resto do script (formato do CSV, chrF, repetição, totais
    por seção) sem modelo e sem traduções reais."""

    def translate(self, text, src_lang, tgt_lang, _reference=None):
        # O loop de avaliação passa a referência em translate_ref, assim a saída do
        # dry-run é sempre igual e o chrF não fica 100.
        words = (_reference or text).split()
        if len(words) > 3:
            words = words[:-1]  # tira a última palavra para simular uma omissão
        return " ".join(words)


def sentence_len_ratio(pred, ref):
    ref_len = max(1, len(ref.split()))
    return round(len(pred.split()) / ref_len, 3)


def evaluate(rows, translator, directions, dry_run):
    results = []
    for r in rows:
        for direction in directions:
            if direction == "pt2tca":
                src_lang, tgt_lang = "por", "tca"
                source_text, reference_text = r["por"], r["tca"]
            else:
                src_lang, tgt_lang = "tca", "por"
                source_text, reference_text = r["tca"], r["por"]

            if dry_run:
                predicted_text = translator.translate(
                    source_text, src_lang, tgt_lang, _reference=reference_text
                )
            else:
                predicted_text = translator.translate(source_text, src_lang, tgt_lang)

            sentence_chrf = sacrebleu.sentence_chrf(
                predicted_text, [reference_text], word_order=2
            ).score

            results.append({
                "source_file": "ticuna_pt_testset.tsv",
                "src_lang": "pt" if src_lang == "por" else "tca",
                "tgt_lang": "pt" if tgt_lang == "por" else "tca",
                "id": r["id"],
                "section": r["section"],
                "paragraph": r["paragraph"],
                "type": r["type"],
                "source_text": source_text,
                "reference_text": reference_text,
                "predicted_text": predicted_text,
                "sentence_chrf": round(sentence_chrf, 2),
                "has_repetition": has_pathological_repetition(predicted_text),
                "length_ratio": sentence_len_ratio(predicted_text, reference_text),
            })
    return results


def add_group_rollups(results):
    """Adiciona group_chrf / group_bleu / group_repeat_pct / group_len_ratio
    por (source_file, src_lang, tgt_lang), no mesmo formato de CSV usado pelo
    build_review_sample.py. Também calcula um resumo por seção, que é gravado
    à parte (section não faz parte do formato comum, então não entra nessas
    4 colunas)."""
    by_cell = defaultdict(list)
    for r in results:
        by_cell[(r["source_file"], r["src_lang"], r["tgt_lang"])].append(r)

    for cell, rows in by_cell.items():
        preds = [r["predicted_text"] for r in rows]
        refs = [[r["reference_text"] for r in rows]]
        group_chrf = sacrebleu.corpus_chrf(preds, refs, word_order=2).score
        group_bleu = sacrebleu.corpus_bleu(preds, refs).score
        repeat_pct = 100.0 * sum(r["has_repetition"] for r in rows) / len(rows)
        len_ratio = sum(r["length_ratio"] for r in rows) / len(rows)
        for r in rows:
            r["group_chrf"] = round(group_chrf, 2)
            r["group_bleu"] = round(group_bleu, 2)
            r["group_repeat_pct"] = round(repeat_pct, 2)
            r["group_len_ratio"] = round(len_ratio, 3)

    by_section = defaultdict(list)
    for r in results:
        by_section[(r["section"], r["src_lang"], r["tgt_lang"])].append(r)
    section_summary = []
    for (section, src, tgt), rows in sorted(by_section.items()):
        preds = [r["predicted_text"] for r in rows]
        refs = [[r["reference_text"] for r in rows]]
        chrf = sacrebleu.corpus_chrf(preds, refs, word_order=2).score
        bleu = sacrebleu.corpus_bleu(preds, refs).score
        repeat_pct = 100.0 * sum(r["has_repetition"] for r in rows) / len(rows)
        section_summary.append({
            "section": section, "src_lang": src, "tgt_lang": tgt,
            "n": len(rows), "chrf": round(chrf, 2), "bleu": round(bleu, 2),
            "repeat_pct": round(repeat_pct, 2),
        })
    return section_summary


def write_results_csv(results, out_path):
    fields = [
        "source_file", "src_lang", "tgt_lang", "id", "section", "paragraph", "type",
        "source_text", "reference_text", "predicted_text", "sentence_chrf",
        "has_repetition", "length_ratio", "group_chrf", "group_bleu",
        "group_repeat_pct", "group_len_ratio",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)


def write_section_summary_csv(section_summary, out_path):
    fields = ["section", "src_lang", "tgt_lang", "n", "chrf", "bleu", "repeat_pct"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(section_summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None,
                         help="Pasta com o modelo CT2 int8 e o tokenizador "
                              "(ex.: a saída do optimize_for_deployment.py). "
                              "Não é necessária com --dry-run.")
    parser.add_argument("--testset-tsv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--section-summary-csv", default=None,
                         help="Caminho opcional para o resumo de chrF/BLEU por seção. "
                              "Padrão: <output-csv>.sections.csv")
    parser.add_argument("--directions", choices=["pt2tca", "tca2pt", "both"], default="both")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true",
                         help="Não carrega o ctranslate2 nem o modelo; roda o resto do "
                              "script com um tradutor falso. Serve para testar o "
                              "próprio script.")
    args = parser.parse_args()

    if not args.dry_run and not args.model_dir:
        raise SystemExit("--model-dir é obrigatório, a não ser com --dry-run.")

    directions = ["pt2tca", "tca2pt"] if args.directions == "both" else [args.directions]

    rows = read_testset_tsv(args.testset_tsv)
    print(f"{len(rows)} linhas válidas (por, tca) carregadas de {args.testset_tsv}")

    if args.dry_run:
        translator = DryRunTranslator()
        print("DRY RUN: usando tradutor falso, nenhum modelo será carregado.")
    else:
        print(f"Carregando modelo CT2 de {args.model_dir} ...")
        translator = Ct2Translator(
            args.model_dir,
            beam_size=args.beam_size,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
            max_len=args.max_len,
        )

    results = evaluate(rows, translator, directions, args.dry_run)
    section_summary = add_group_rollups(results)

    write_results_csv(results, args.output_csv)
    section_csv = args.section_summary_csv or (str(args.output_csv) + ".sections.csv")
    write_section_summary_csv(section_summary, section_csv)

    for (src_lang, tgt_lang), rows_ in (
        (k, [r for r in results if (r["src_lang"], r["tgt_lang"]) == k])
        for k in sorted({(r["src_lang"], r["tgt_lang"]) for r in results})
    ):
        chrf = rows_[0]["group_chrf"]
        bleu = rows_[0]["group_bleu"]
        repeat_pct = rows_[0]["group_repeat_pct"]
        print(f"{src_lang}->{tgt_lang}: n={len(rows_)}  chrF++={chrf}  "
              f"BLEU={bleu}  repeat%={repeat_pct}")

    print(f"\nResultados por exemplo gravados em: {args.output_csv}")
    print(f"Resumo por seção gravado em: {section_csv}")


if __name__ == "__main__":
    main()
