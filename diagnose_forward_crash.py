"""
Script de diagnóstico para o erro decoder_input_ids/decoder_inputs_embeds.
Chama o modelo direto com um batch real, fora do Trainer/accelerate/autocast.
O caso que falhar mostra onde está o problema.

Executar:
    python diagnose_forward_crash.py
"""

import sys
import traceback

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config
from text_utils import normalize_text  # noqa: F401 (importado só pelos efeitos colaterais)

# Reaproveita o mesmo Dataset/Collator do treino sem importar o módulo
# 03_train inteiro (o nome começa com número, o que complica o import).
import importlib.util
spec = importlib.util.spec_from_file_location("train_mod", "03_train.py")
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)


def try_forward(model, batch, label, use_autocast, no_grad):
    print(f"\n{'='*70}\nAttempt: {label}\n{'='*70}")
    try:
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else _nullcontext()
        grad_ctx = torch.no_grad() if no_grad else _nullcontext()
        with ctx, grad_ctx:
            out = model(**batch)
        print(f"SUCESSO: loss {out.loss.item() if out.loss is not None else 'N/A'}")
        return True
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def main():
    print("Carregando tokenizador e modelo...")
    tokenizer = AutoTokenizer.from_pretrained(config.EXTENDED_TOKENIZER_DIR)

    lang_code_map = train_mod.build_lang_code_map()
    ds = train_mod.TranslationDataset(config.TRAIN_JSONL, tokenizer, lang_code_map, config.MAX_LEN)
    collate = train_mod.Collator(pad_id=tokenizer.pad_token_id)
    batch_cpu = collate([ds[0], ds[1], ds[2]])

    print("\nConteúdo do batch:")
    for k, v in batch_cpu.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")

    results = {}

    def fresh_model():
        m = AutoModelForSeq2SeqLM.from_pretrained(config.BASE_MODEL)
        old_vocab = len(m.get_input_embeddings().weight)
        m.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
        return m, old_vocab

    def run_case(label, apply_custom_init, enable_checkpointing, use_autocast, train_mode):
        model, old_vocab = fresh_model()
        if apply_custom_init:
            train_mod.init_new_embeddings_as_mean(model, old_vocab)
        if enable_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        model = model.to("cuda")
        batch = {k: v.to("cuda") for k, v in batch_cpu.items()}
        model.train() if train_mode else model.eval()
        results[label] = try_forward(model, batch, label, use_autocast=use_autocast,
                                      no_grad=not train_mode)
        del model
        torch.cuda.empty_cache()

    # Caso base (já confirmado que funciona; roda de novo para comparar)
    run_case("base: sem init customizado, sem checkpointing", False, False, True, True)
    # Só a inicialização dos embeddings com ruído
    run_case("+ init_new_embeddings_as_mean", True, False, True, True)
    # Só o gradient checkpointing
    run_case("+ gradient_checkpointing_enable()", False, True, True, True)
    # Os dois juntos, igual ao 03_train.py com --freeze-steps 0
    run_case("+ OS DOIS (igual ao 03_train.py com --freeze-steps 0)", True, True, True, True)

    print(f"\n{'='*70}\nSummary\n{'='*70}")
    for label, ok in results.items():
        print(f"{'OK      ' if ok else 'FALHOU  '} -- {label}")
    print("\nA linha que falhar aqui indica a causa.")


if __name__ == "__main__":
    main()
