"""
Diagnóstico da saída vazia no translate_cpu.py: confere se o vocabulário do
modelo convertido para CT2 bate com o tokenizador HF original. Verifica se o
tca_Latn continua com o mesmo id depois da conversão e se os tamanhos dos
vocabulários são iguais. Um vocabulário redimensionado e com embeddings
compartilhados, como neste modelo, é o tipo de caso em que a conversão
HF->CT2 pode dar errado sem avisar.

Executar:
    python diagnose_ct2_vocab.py --hf-checkpoint nllb_tca_checkpoints/final --ct2-model nllb_tca_ct2
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-checkpoint", type=str, default="nllb_tca_checkpoints/final")
    parser.add_argument("--ct2-model", type=str, default="nllb_tca_ct2")
    args = parser.parse_args()

    hf_dir = Path(args.hf_checkpoint)
    ct2_dir = Path(args.ct2_model)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    hf_vocab_size = len(tokenizer)
    tca_id_hf = tokenizer.convert_tokens_to_ids("tca_Latn")
    por_id_hf = tokenizer.convert_tokens_to_ids("por_Latn")
    print(f"Tokenizador HF: vocab_size={hf_vocab_size}")
    print(f"  'tca_Latn' -> id {tca_id_hf}")
    print(f"  'por_Latn' -> id {por_id_hf}")
    if tca_id_hf == tokenizer.unk_token_id:
        print("  !! 'tca_Latn' vira o id UNK no próprio tokenizador HF; "
              "o problema é anterior à conversão para CT2.")

    ct2_vocab_path = ct2_dir / "shared_vocabulary.json"
    if not ct2_vocab_path.exists():
        # Algumas versões/tipos de modelo do CT2 usam arquivos separados de vocabulário de origem e destino
        alt = list(ct2_dir.glob("*vocabulary*.json"))
        print(f"\nshared_vocabulary.json não encontrado. Outros arquivos de vocabulário: {alt}")
        return

    with open(ct2_vocab_path, encoding="utf-8") as f:
        ct2_vocab = json.load(f)
    ct2_vocab_size = len(ct2_vocab)
    print(f"\nModelo CT2: vocab_size={ct2_vocab_size}")

    if ct2_vocab_size != hf_vocab_size:
        print(f"  !! DIFERENÇA: vocabulário CT2 ({ct2_vocab_size}) != vocabulário do tokenizador HF "
              f"({hf_vocab_size}). Só isso já pode gerar saída vazia ou sem sentido, se os ids "
              f"mudaram de posição na conversão.")
    else:
        print("  OK: os tamanhos dos vocabulários batem.")

    # ct2_vocab costuma ser uma lista de tokens indexada pela posição, ou um dict;
    # trata os dois casos
    if isinstance(ct2_vocab, list):
        tca_id_ct2 = ct2_vocab.index("tca_Latn") if "tca_Latn" in ct2_vocab else None
        por_id_ct2 = ct2_vocab.index("por_Latn") if "por_Latn" in ct2_vocab else None
    else:
        tca_id_ct2 = ct2_vocab.get("tca_Latn")
        por_id_ct2 = ct2_vocab.get("por_Latn")

    print(f"\n'tca_Latn' no vocabulário CT2: {'encontrado no id ' + str(tca_id_ct2) if tca_id_ct2 is not None else 'AUSENTE'}")
    print(f"'por_Latn' no vocabulário CT2: {'encontrado no id ' + str(por_id_ct2) if por_id_ct2 is not None else 'AUSENTE'}")

    if tca_id_ct2 is None:
        print("\n!! 'tca_Latn' não existe no vocabulário convertido para CT2.")
        print("   Isso explicaria a saída vazia: target_prefix=['tca_Latn'] não")
        print("   corresponde a nenhum token. O problema estaria na conversão,")
        print("   e não no translate_cpu.py.")
    elif tca_id_hf != tca_id_ct2:
        print(f"\n!! IDs DIFERENTES: 'tca_Latn' é o id {tca_id_hf} no tokenizador HF e "
              f"o id {tca_id_ct2} no vocabulário CT2. Se as linhas dos embeddings foram "
              f"reordenadas na conversão, isso gera exatamente a saída vazia ou sem "
              f"sentido que estamos vendo.")
    else:
        print("\nOK: o id de 'tca_Latn' é igual no HF e no CT2; "
              "essa causa está descartada.")


if __name__ == "__main__":
    main()
