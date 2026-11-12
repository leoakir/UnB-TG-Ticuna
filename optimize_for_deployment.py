"""
Converte o checkpoint treinado do NLLB para o formato CTranslate2 com
quantização int8, que é o caminho padrão para rodar modelos NLLB/M2M100 só
na CPU. Para um modelo deste tamanho, o resultado costuma ser ~4x menor em
disco (int8 em vez de float32) e várias vezes mais rápido na CPU, com pouca
perda de qualidade.

Isso importa porque o modelo vai rodar nos computadores dos estudantes, que
na maioria não têm GPU. A RTX 3080 só é necessária para treinar.

Dependências: pip install ctranslate2

Executar:
    python optimize_for_deployment.py
    python optimize_for_deployment.py --checkpoint nllb_tca_checkpoints/final --benchmark
"""

import argparse
import shutil
import time
from pathlib import Path

import config


def get_dir_size_mb(path):
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / (1024 * 1024)


def run_conversion(checkpoint_dir, output_dir, quantization):
    print(f"Converting {checkpoint_dir} -> {output_dir} (quantization={quantization})...")
    # Usa a API do Python direto, em vez de chamar o ct2-transformers-converter
    # por subprocess. No Windows, subprocess.run() com lista de argumentos nem
    # sempre encontra os comandos instalados pelo pip (dá WinError 2 mesmo quando
    # o comando funciona no terminal). A CLI só chama essa mesma API, então o
    # resultado é igual e funciona do mesmo jeito em qualquer sistema.
    from ctranslate2.converters import TransformersConverter
    converter = TransformersConverter(str(checkpoint_dir))
    try:
        converter.convert(str(output_dir), quantization=quantization, force=True)
    except Exception as e:
        raise SystemExit(f"A conversão falhou: {e}\nCausa comum: faltou 'pip install ctranslate2' "
                          f"ou a pasta do checkpoint não tem um modelo HF válido.")
    print("Conversão concluída.")


def benchmark(checkpoint_dir, ct2_dir, num_samples=20):
    """Compara tamanho em disco e velocidade na CPU entre HF float32 e CT2 int8,
    usando frases reais do teste."""
    import json
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import ctranslate2

    import importlib.util
    spec = importlib.util.spec_from_file_location("train_mod", "03_train.py")
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    lang_code_map = train_mod.build_lang_code_map()

    examples = []
    with open(config.TEST_JSONL, encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if ex["src_lang"] == "pt" and ex["tgt_lang"] == "tca":
                examples.append(ex)
            if len(examples) >= num_samples:
                break

    print(f"\n{'='*60}\nBenchmark ({num_samples} frases reais pt->tca, CPU)\n{'='*60}")

    hf_size = get_dir_size_mb(checkpoint_dir)
    ct2_size = get_dir_size_mb(ct2_dir)
    print(f"Tamanho em disco: checkpoint HF = {hf_size:.0f} MB, CT2 int8 = {ct2_size:.0f} MB "
          f"({hf_size/ct2_size:.1f}x menor)")

    print("\nCarregando o modelo HF na CPU (referência lenta, sem otimização)...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
    hf_model.eval()

    def lang_id(lang_key):
        return tokenizer.convert_tokens_to_ids(lang_code_map[lang_key])

    start = time.time()
    for ex in examples:
        enc = tokenizer(ex["src_text"], return_tensors="pt")
        input_ids = torch.cat([torch.tensor([[lang_id("pt")]]), enc["input_ids"]], dim=1)
        attention_mask = torch.cat([torch.tensor([[1]]), enc["attention_mask"]], dim=1)
        with torch.no_grad():
            hf_model.generate(input_ids=input_ids, attention_mask=attention_mask,
                               forced_bos_token_id=lang_id("tca"), max_length=config.MAX_LEN,
                               num_beams=1)
    hf_time = time.time() - start
    print(f"HF (float32, CPU): {hf_time:.1f}s no total, {hf_time/len(examples)*1000:.0f}ms/frase")

    print("\nCarregando o modelo CT2 int8...")
    translator = ctranslate2.Translator(str(ct2_dir), device="cpu")
    start = time.time()
    for ex in examples:
        src_tokens = [lang_code_map["pt"]] + tokenizer.tokenize(ex["src_text"])
        translator.translate_batch([src_tokens], target_prefix=[[lang_code_map["tca"]]],
                                    max_decoding_length=config.MAX_LEN, beam_size=1)
    ct2_time = time.time() - start
    print(f"CT2 (int8, CPU):    {ct2_time:.1f}s no total, {ct2_time/len(examples)*1000:.0f}ms/frase")
    print(f"\nCT2 é {hf_time/ct2_time:.1f}x mais rápido na CPU")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(config.MODEL_OUTPUT_DIR / "final"),
                         help="Pasta do checkpoint HF a converter.")
    parser.add_argument("--output-dir", type=str, default="./nllb_tca_ct2",
                         help="Pasta onde salvar o modelo CTranslate2.")
    parser.add_argument("--quantization", type=str, default="int8",
                         choices=["int8", "int8_float32", "int16", "float16", "float32"],
                         help="int8 é o padrão indicado para CPU em máquinas simples: menor "
                              "tamanho, mais rápido e com pouca perda de qualidade "
                              "para um modelo deste tamanho.")
    parser.add_argument("--benchmark", action="store_true",
                         help="Depois de converter, compara velocidade e tamanho com o "
                              "modelo HF original usando frases reais.")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not checkpoint_dir.exists():
        raise SystemExit(f"Checkpoint não encontrado: {checkpoint_dir}")

    run_conversion(checkpoint_dir, output_dir, args.quantization)

    # Copia também os arquivos do tokenizador. O translate_cpu.py precisa
    # deles para tokenizar a entrada e decodificar a saída; o model.bin do CT2
    # só tem os pesos da rede.
    print("\nCopiando os arquivos do tokenizador para a pasta do modelo CT2...")
    for fname in ["sentencepiece.bpe.model", "tokenizer_config.json", "special_tokens_map.json",
                  "added_tokens.json", "tokenizer.json"]:
        src = checkpoint_dir / fname
        if src.exists():
            shutil.copy(src, output_dir / fname)
            print(f"  copied {fname}")

    lang_detector_src = Path("lang_detector.json")
    if lang_detector_src.exists():
        shutil.copy(lang_detector_src, output_dir / "lang_detector.json")
        print(f"  lang_detector.json copiado (se não aparecer aqui, rode antes o "
              f"build_lang_detector.py)")

    print(f"\nPronto. Modelo para uso em: {output_dir}")
    print(f"Copie a pasta inteira para a máquina de uso; não precisa de mais nada "
          f"(nem GPU, nem os arquivos do treino).")

    if args.benchmark:
        benchmark(checkpoint_dir, output_dir)


if __name__ == "__main__":
    main()
