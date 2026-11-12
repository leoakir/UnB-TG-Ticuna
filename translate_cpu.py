"""
Tradução em CPU com o modelo convertido para CTranslate2.

Arquivos necessários:
    translate_cpu.py (este)
    pasta do modelo CT2 (padrão: nllb_tca_ct2)
    text_utils.py

Dependências: pip install ctranslate2 transformers

Executar:
    python translate_cpu.py --model ./nllb_tca_ct2
        Modo interativo, detecta o idioma da frase sozinho (98,5% de acerto nos testes)

    python translate_cpu.py --model ./nllb_tca_ct2 --text "Bom dia" --src pt --tgt tca
        Traduz uma frase só
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import ctranslate2
from transformers import AutoTokenizer

from text_utils import normalize_text  # mesma normalização do treino

LANG_CODES = {"pt": "por_Latn", "es": "spa_Latn", "tca": "tca_Latn"}


def char_ngrams(text, n_sizes=(2, 3)):
    text = text.lower()
    grams = []
    for n in n_sizes:
        grams.extend(text[i:i+n] for i in range(len(text) - n + 1))
    return grams


class LangDetector:
    """Usa o perfil do build_lang_detector.py para classificar pt ou tca com
    Naive Bayes sobre n-gramas de caracteres. Foi treinado com os dados do
    projeto, já que as ferramentas comuns de detecção de idioma não têm Ticuna."""

    def __init__(self, profile_path):
        with open(profile_path, encoding="utf-8") as f:
            profile = json.load(f)
        self.pt_counts = profile["pt_counts"]
        self.tca_counts = profile["tca_counts"]
        self.pt_total = profile["pt_total"]
        self.tca_total = profile["tca_total"]
        self.vocab_size = profile["vocab_size"]

    def _score(self, text, counts, total):
        log_prob = 0.0
        for g in char_ngrams(text):
            log_prob += math.log((counts.get(g, 0) + 1) / (total + self.vocab_size))
        return log_prob

    def detect(self, text):
        pt_score = self._score(text, self.pt_counts, self.pt_total)
        tca_score = self._score(text, self.tca_counts, self.tca_total)
        return "pt" if pt_score >= tca_score else "tca"


class Translator:
    def __init__(self, model_dir, threads=4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # inter_threads/intra_threads fazem diferença em máquinas com poucos
        # núcleos; diminua se for rodar junto com outros programas.
        self.translator = ctranslate2.Translator(model_dir, device="cpu",
                                                   inter_threads=1, intra_threads=threads)

    def translate(self, text, src_lang, tgt_lang, beam_size=4, max_length=128, debug=False):
        if src_lang not in LANG_CODES or tgt_lang not in LANG_CODES:
            raise ValueError(f"Código de idioma desconhecido. Use: {list(LANG_CODES)}")

        text = normalize_text(text)

        # Tokeniza com tokenizer() e converte os ids de volta para tokens,
        # igual ao treino e à avaliação.
        input_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        # A sequência é [token do idioma] + subpalavras + [EOS], como no treino
        # (03_train.py -> TranslationDataset._encode).
        src_tokens = ([LANG_CODES[src_lang]] + self.tokenizer.convert_ids_to_tokens(input_ids)
                       + [self.tokenizer.eos_token])

        if debug:
            print(f"  [debug] tokens de entrada: {src_tokens}", flush=True)
        start = time.time()

        results = self.translator.translate_batch(
            [src_tokens],
            target_prefix=[[LANG_CODES[tgt_lang]]],
            beam_size=beam_size,
            max_decoding_length=max_length,
            # evita que frases longas entrem em loop de repetição
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )
        if debug:
            print(f"  [debug] {time.time()-start:.2f}s, "
                  f"tokens de saída: {results[0].hypotheses[0]}", flush=True)

        output_tokens = results[0].hypotheses[0][1:]  # remove o token de idioma de destino forçado
        output_ids = self.tokenizer.convert_tokens_to_ids(output_tokens)
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)


DIRECTIONS = [("pt", "tca"), ("tca", "pt")]


def choose_direction():
    print("Detector de idioma não encontrado. Escolha a direção da tradução:")
    for i, (src, tgt) in enumerate(DIRECTIONS, 1):
        print(f"  {i}) {src} -> {tgt}")
    while True:
        choice = input("> ").strip()
        if choice in ("1", "2"):
            return DIRECTIONS[int(choice) - 1]
        print("Digite uma das opções.")


def interactive_loop(translator, detector, args):
    if detector is None:
        # Só é usado se faltar o lang_detector.json. O normal é a detecção
        # automática abaixo; assim o script continua funcionando mesmo sem o arquivo.
        src_lang, tgt_lang = choose_direction()
        print(f"\nTraduzindo {src_lang} -> {tgt_lang}. Digite uma frase ou '0' para sair.\n")
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text.lower() in ("sair", "exit", "quit", "0"):
                break
            translation = translator.translate(text, src_lang, tgt_lang,
                                                 beam_size=args.beam_size, debug=args.debug)
            print(f'[{src_lang}]: "{text}"')
            print(f'[{tgt_lang}]: "{translation}"\n')
        return

    print("Digite uma frase em português (pt) ou Ticuna (tca) para traduzir. "
          "Digite '0' para sair.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("sair", "exit", "quit", "0"):
            break

        src_lang = detector.detect(text)
        tgt_lang = "tca" if src_lang == "pt" else "pt"
        translation = translator.translate(text, src_lang, tgt_lang,
                                             beam_size=args.beam_size, debug=args.debug)
        print(f'[{src_lang}]: "{text}"')
        print(f'[{tgt_lang}]: "{translation}"\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         help="Pasta do modelo CT2 (gerada pelo optimize_for_deployment.py).")
    parser.add_argument("--text", type=str, default=None,
                         help="Traduz uma frase e encerra, em vez do modo interativo. Precisa de --src/--tgt.")
    parser.add_argument("--src", type=str, default="pt", choices=list(LANG_CODES),
                         help="Só usado com --text (ignorado se --auto for usado).")
    parser.add_argument("--tgt", type=str, default="tca", choices=list(LANG_CODES),
                         help="Só usado com --text (ignorado se --auto for usado).")
    parser.add_argument("--auto", action="store_true",
                         help="Com --text: detecta pt ou tca em vez de usar --src/--tgt. Precisa do lang_detector.json.")
    parser.add_argument("--detector", type=str, default=None,
                         help="Caminho do detector gerado pelo build_lang_detector.py. Padrão: lang_detector.json "
                              "dentro da pasta de --model.")
    parser.add_argument("--beam-size", type=int, default=1,
                         help="1 = mais rápido na CPU, bom para o uso normal. 4 ou 5 dá mais qualidade em frases "
                              "importantes, mas é mais lento.")
    parser.add_argument("--threads", type=int, default=4,
                         help="Threads de CPU. Diminua (ex.: 2) se a tradução estiver disputando a CPU com outros programas.")
    parser.add_argument("--debug", action="store_true",
                         help="Mostra os tokens de entrada/saída e o tempo de cada tradução. Só para depuração.")
    args = parser.parse_args()

    print(f"Carregando o modelo {args.model}...", flush=True)
    translator = Translator(args.model, threads=args.threads)

    detector = None
    detector_path = Path(args.detector) if args.detector else Path(args.model) / "lang_detector.json"
    if detector_path.exists():
        detector = LangDetector(detector_path)
        print(f"Detecção automática de idioma ativada ({detector_path}).", flush=True)
    print("Pronto.\n", flush=True)

    if args.text:
        src_lang, tgt_lang = args.src, args.tgt
        if args.auto:
            if detector is None:
                raise SystemExit(f"--auto precisa do perfil de detecção em {detector_path} "
                                  f"(gere com o build_lang_detector.py).")
            src_lang = detector.detect(args.text)
            tgt_lang = "tca" if src_lang == "pt" else "pt"
        translation = translator.translate(args.text, src_lang, tgt_lang,
                                            beam_size=args.beam_size, debug=args.debug)
        print(f'[{src_lang}]: "{args.text}"')
        print(f'[{tgt_lang}]: "{translation}"')
    else:
        interactive_loop(translator, detector, args)


if __name__ == "__main__":
    main()
