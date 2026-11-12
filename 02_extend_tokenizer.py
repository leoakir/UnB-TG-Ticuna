"""
Passo 2: estender o tokenizador do NLLB para o Ticuna (tca).

O tokenizador do NLLB é um modelo SentencePiece único para os 200 idiomas.
Como o Ticuna não está entre eles, as palavras seriam quebradas em muitos
pedaços pequenos sem sentido. Este passo:

  1. Treina um modelo SentencePiece pequeno no texto monolíngue em Ticuna
     gerado no passo 1.
  2. Junta as peças novas ao modelo do NLLB. As peças que o NLLB já tem
     (subpalavras de português e espanhol, pontuação etc.) não são
     removidas nem reordenadas; só adicionamos.
  3. Registra um token novo de idioma (tca_Latn).

O 03_train.py monta as sequências na mão (token do idioma no início, eos no
fim), o que funciona igual para idiomas que o modelo já conhece e para o
idioma novo.

Dependências: pip install transformers sentencepiece protobuf

Executar:
    python 02_extend_tokenizer.py
"""

import os
import re

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as spm_pb2
from transformers import AutoTokenizer

import config


def train_new_spm_pieces(text_path, vocab_size, model_prefix):
    """A instalação do sentencepiece veio sem os charsmap pré-compilados do
    package_data. Por isso passamos um normalization_rule_tsv vazio: assim
    as regras são compiladas no treino em vez de carregadas do package_data."""
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    empty_rules_path = str(config.WORK_DIR / "identity_rules.tsv")
    if not os.path.exists(empty_rules_path):
        """Um arquivo vazio também dá erro, e mapear um caractere para ele mesmo
        também. Então criamos uma regra entre dois caracteres de uso privado,
        que nunca aparecem em texto normal. Isso satisfaz o sentencepiece sem
        mudar nada no texto tokenizado."""
        with open(empty_rules_path, "w", encoding="utf-8") as f:
            f.write("E000\tE001\n")

    # O vocab_size pedido (config.NEW_SPM_VOCAB_SIZE) é só uma meta. Com um
    # corpus pequeno, o treino unigram só consegue gerar as peças que o corpus
    # permite. Para não ter que ajustar o número toda vez que o corpus muda, o
    # script tenta de novo com o máximo que o sentencepiece informa.
    attempted = vocab_size
    while True:
        try:
            spm.SentencePieceTrainer.train(
                input=str(text_path),
                model_prefix=model_prefix,
                vocab_size=attempted,
                model_type="unigram",
                character_coverage=0.9995,
                # character_coverage: perto de 1.0 para línguas com muitos diacríticos
                input_sentence_size=0,
                shuffle_input_sentence=True,
                normalization_rule_tsv=empty_rules_path,
                pad_id=-1, unk_id=0, bos_id=-1, eos_id=-1,
            )
            break
        except RuntimeError as e:
            msg = str(e)
            match = re.search(r"<=\s*(\d+)", msg)
            if "Vocabulary size too high" in msg and match:
                capped = int(match.group(1))
                print(f"vocab_size={attempted} é maior do que o corpus permite, "
                      f"tentando de novo com o máximo informado: {capped}")
                attempted = capped
                continue
            raise

    if attempted != vocab_size:
        print(f"OBS: vocabulário tca treinado com {attempted} peças "
              f"(pedido: {vocab_size}), limitado pelo tamanho do corpus "
              f"monolíngue. É normal para um corpus pequeno, não é erro.")

    return f"{model_prefix}.model"


def merge_into_nllb_spm(nllb_spm_path, new_spm_path, out_path):
    base = spm_pb2.ModelProto()
    with open(nllb_spm_path, "rb") as f:
        base.ParseFromString(f.read())

    existing_pieces = {p.piece for p in base.pieces}

    new_model = spm_pb2.ModelProto()
    with open(new_spm_path, "rb") as f:
        new_model.ParseFromString(f.read())

    added = 0
    # Score abaixo do mínimo do vocabulário base, para as peças novas não
    # passarem na frente das subpalavras originais do NLLB na tokenização.
    min_score = min(p.score for p in base.pieces)
    for i, piece in enumerate(new_model.pieces):
        if piece.piece in existing_pieces:
            continue
        if piece.type != spm_pb2.ModelProto.SentencePiece.NORMAL:
            continue  # pula os tokens especiais/de controle do modelo novo
        new_piece = base.pieces.add()
        new_piece.piece = piece.piece
        new_piece.score = min_score - 1.0 - i * 1e-6  # scores sempre decrescentes
        new_piece.type = spm_pb2.ModelProto.SentencePiece.NORMAL
        existing_pieces.add(piece.piece)
        added += 1

    with open(out_path, "wb") as f:
        f.write(base.SerializeToString())

    print(f"{added} peças novas adicionadas (a base tinha {len(base.pieces) - added}, "
          f"agora o vocabulário tem {len(base.pieces)}).")
    return added


def main():
    config.EXTENDED_TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    work_prefix = str(config.WORK_DIR / "tca_spm")

    print("Treinando o SentencePiece no texto monolíngue em Ticuna...")
    new_spm_path = train_new_spm_pieces(
        config.TCA_MONOLINGUAL_TXT, config.NEW_SPM_VOCAB_SIZE, work_prefix
    )

    print("Carregando o tokenizador base do NLLB...")
    base_tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    base_spm_path = base_tokenizer.vocab_file  # caminho do sentencepiece.bpe.model
    print(f"Tamanho do vocabulário base: {len(base_tokenizer)}")

    merged_spm_path = str(config.EXTENDED_TOKENIZER_DIR / "sentencepiece.bpe.model")
    merge_into_nllb_spm(base_spm_path, new_spm_path, merged_spm_path)

    print("Reconstruindo o tokenizador HF a partir do SentencePiece combinado...")
    new_tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL, vocab_file=merged_spm_path
    )

    print(f"Adicionando o token de idioma: {config.NEW_LANG_CODE}")
    num_added = new_tokenizer.add_tokens([config.NEW_LANG_CODE], special_tokens=True)
    new_lang_id = new_tokenizer.convert_tokens_to_ids(config.NEW_LANG_CODE)
    print(f"'{config.NEW_LANG_CODE}' -> token id {new_lang_id} ({num_added} token especial adicionado)")

    new_tokenizer.save_pretrained(config.EXTENDED_TOKENIZER_DIR)

    # Salva o id para o 03_train.py não precisar calcular de novo.
    with open(config.EXTENDED_TOKENIZER_DIR / "new_lang_id.txt", "w") as f:
        f.write(str(new_lang_id))

    print(f"\nTamanho final do tokenizador: {len(new_tokenizer)}")
    print(f"Tokenizador salvo em: {config.EXTENDED_TOKENIZER_DIR}")

    # Teste rápido
    sample = open(config.TCA_MONOLINGUAL_TXT, encoding="utf-8").readline().strip()
    if sample:
        ids = new_tokenizer.encode(sample)
        print(f"\nFrase tca: {sample[:80]!r}")
        print(f"Tokenizada em {len(ids)} ids: {ids[:20]}...")


if __name__ == "__main__":
    main()
