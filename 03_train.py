"""
Passo 3: fine-tuning do facebook/nllb-200-distilled-600M em pt/es <-> tca.

Configurado para i7-13700K + RTX 3080 10GB:
    - Fine-tuning completo. O modelo de 600M cabe em 10GB com fp16/bf16 e
      gradient checkpointing, então não precisa de LoRA/QLoRA.
    - Batch pequeno por GPU + acumulação de gradiente, para simular um batch
      maior sem estourar a VRAM.
    - Adam 8-bit opcional (bitsandbytes), para sobrar VRAM e usar um batch
      maior em vez de mais passos de acumulação.

As sequências são montadas na mão no formato do NLLB
([lang_id] + subpalavras + [eos]) porque o tca não está na lista de idiomas
do tokenizador.

Dependências: pip install transformers accelerate sentencepiece sacrebleu bitsandbytes

Executar:
    python 03_train.py
    python 03_train.py --use-8bit-adam --per-device-batch-size 4 --grad-accum 8
"""

import argparse
import inspect
import json

import sacrebleu
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

import config


class TranslationDataset(Dataset):
    """Lê um arquivo jsonl e monta as sequências no formato do NLLB na hora.
    Funciona para qualquer código de idioma, inclusive o tca_Latn, porque não
    depende da lista de idiomas do tokenizador."""

    def __init__(self, jsonl_path, tokenizer, lang_code_map, max_len):
        self.examples = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.lang_code_map = lang_code_map  # ex.: {"pt": "por_Latn", "tca": "tca_Latn", ...}
        self.max_len = max_len
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.examples)

    def _lang_id(self, lang_key):
        code = self.lang_code_map[lang_key]
        tid = self.tokenizer.convert_tokens_to_ids(code)
        if tid == self.tokenizer.unk_token_id:
            raise ValueError(f"Código de idioma '{code}' não está no vocabulário do tokenizador.")
        return tid

    def _encode(self, text, lang_key):
        ids = self.tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                     max_length=self.max_len - 2)
        return [self._lang_id(lang_key)] + ids + [self.eos_id]

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_ids = self._encode(ex["src_text"], ex["src_lang"])
        label_ids = self._encode(ex["tgt_text"], ex["tgt_lang"])
        return {"input_ids": input_ids, "labels": label_ids}


class Collator:
    """Junta as frases em batches, com padding só até a maior frase de cada
    batch. É uma classe, e não uma função dentro de outra função, porque no
    Windows o código precisa ser serializado para os processos auxiliares, e
    funções aninhadas não podem ser serializadas."""


    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        pad_id = self.pad_id
        max_in = max(len(b["input_ids"]) for b in batch)
        max_out = max(len(b["labels"]) for b in batch)

        input_ids, attn_mask, labels = [], [], []
        for b in batch:
            ids = b["input_ids"]
            pad_len = max_in - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            attn_mask.append([1] * len(ids) + [0] * pad_len)

            lab = b["labels"]
            pad_len_l = max_out - len(lab)
            # -100 para o padding não entrar na loss
            labels.append(lab + [-100] * pad_len_l)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def init_new_embeddings_as_mean(model, old_num_tokens, noise_std=0.02):
    """Inicializa os vetores das ~6.500 subpalavras novas do Ticuna.

    Por padrão, resize_token_embeddings() preenche esses vetores com valores
    aleatórios. Aqui eles começam perto da média dos vetores que o modelo já
    tem. Mas se todos fossem exatamente a média, seriam idênticos e o modelo
    não conseguiria diferenciá-los no começo (a loss inicial ficou pior que um
    chute aleatório). Por isso somamos um ruído gaussiano pequeno a cada vetor,
    proporcional à variação de cada dimensão."""
    with torch.no_grad():
        embed = model.get_input_embeddings()
        old_embeds = embed.weight[:old_num_tokens]
        mean_vec = old_embeds.mean(dim=0)
        std_vec = old_embeds.std(dim=0)
        num_new = embed.weight.shape[0] - old_num_tokens

        noise = torch.randn(num_new, embed.weight.shape[1],
                             device=embed.weight.device, dtype=embed.weight.dtype)
        noise = noise * std_vec.to(embed.weight.dtype) * noise_std
        embed.weight[old_num_tokens:] = mean_vec.to(embed.weight.dtype) + noise

        if not model.config.tie_word_embeddings:
            out_embed = model.get_output_embeddings()
            if out_embed is not None:
                out_noise = torch.randn(num_new, out_embed.weight.shape[1],
                                         device=out_embed.weight.device, dtype=out_embed.weight.dtype)
                out_noise = out_noise * std_vec.to(out_embed.weight.dtype) * noise_std
                out_embed.weight[old_num_tokens:] = mean_vec.to(out_embed.weight.dtype) + out_noise


def build_lang_code_map():
    m = dict(config.NLLB_LANG_CODE)
    m[config.NEW_LANG_KEY] = config.NEW_LANG_CODE
    return m


class UnfreezeBackboneCallback(TrainerCallback):
    """Treino em duas fases. Nos primeiros `freeze_steps` passos só as camadas
    de embedding são treinadas e o resto do modelo fica congelado. Assim os
    embeddings novos do tca, que começam quase aleatórios, se ajustam antes de
    mexer nos pesos que já sabem português, espanhol etc. Depois disso o
    modelo inteiro é liberado e o treino continua normalmente, com o mesmo
    otimizador e o mesmo agendamento de LR, sem recarregar checkpoint.

    O gradient checkpointing fica desligado durante a fase congelada e volta
    quando o modelo é liberado. Com quase tudo congelado, esta versão do
    transformers dá um erro em que decoder_input_ids e decoder_inputs_embeds
    ficam definidos ao mesmo tempo (ValueError: "cannot specify both...").
    Desligar o checkpointing só nessa fase evita o erro e mantém a economia de
    memória na fase longa."""

    def __init__(self, model, freeze_steps, use_grad_checkpointing):
        self.model = model
        self.freeze_steps = freeze_steps
        self.use_grad_checkpointing = use_grad_checkpointing
        self._frozen = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.freeze_steps > 0 and state.global_step < self.freeze_steps:
            self._freeze_backbone()

    def on_step_end(self, args, state, control, **kwargs):
        if self._frozen and state.global_step >= self.freeze_steps:
            self._unfreeze_all()

    def _embedding_param_ids(self):
        ids = set()
        for p in self.model.get_input_embeddings().parameters():
            ids.add(id(p))
        out_embed = self.model.get_output_embeddings()
        if out_embed is not None:
            for p in out_embed.parameters():
                ids.add(id(p))
        return ids

    def _freeze_backbone(self):
        embed_ids = self._embedding_param_ids()
        frozen_count = 0
        for p in self.model.parameters():
            if id(p) not in embed_ids:
                p.requires_grad = False
                frozen_count += 1
        self._frozen = True
        if self.use_grad_checkpointing:
            self.model.gradient_checkpointing_disable()
            print("[UnfreezeBackboneCallback] Gradient checkpointing desligado durante a "
                  "fase congelada (evita o conflito entre decoder_input_ids e "
                  "decoder_inputs_embeds com camadas congeladas).")
        print(f"[UnfreezeBackboneCallback] Modelo congelado ({frozen_count} tensores); "
              f"só os embeddings treinam nos primeiros {self.freeze_steps} passos.")

    def _unfreeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = True
        self._frozen = False
        if self.use_grad_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        print(f"[UnfreezeBackboneCallback] Passo {self.freeze_steps}: "
              f"modelo liberado, fine-tuning completo a partir daqui"
              f"{' (gradient checkpointing ligado de novo)' if self.use_grad_checkpointing else ''}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)  # batch efetivo = 32
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--use-8bit-adam", action="store_true",
                         help="Usa o Adam 8-bit do bitsandbytes para liberar VRAM para um batch maior.")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--dataloader-workers", type=int, default=0,
                         help="Processos do DataLoader. Padrão 0 (carrega tudo no processo principal), "
                              "porque o 'spawn' do Windows costuma falhar com Dataset/collator "
                              "personalizados. Pode aumentar (ex.: 4) no Linux/macOS, ou no "
                              "Windows depois de testar.")
    parser.add_argument("--no-grad-checkpointing", action="store_true",
                         help="Desliga o gradient checkpointing. Usa mais VRAM, mas os passos ficam "
                              "bem mais rápidos. O NLLB-600M costuma caber numa placa de 10GB "
                              "mesmo sem ele. Acompanhe o nvidia-smi e tire esta opção "
                              "se chegar perto de 10GB ou der OOM.")
    parser.add_argument("--eval-steps", type=int, default=250,
                         help="A cada quantos passos rodar a avaliação e salvar checkpoint. "
                              "Antes era 200, mas com milhares de passos isso dava dezenas "
                              "de avaliações com geração, que tomam bastante tempo.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                         help="Pasta de um checkpoint em MODEL_OUTPUT_DIR "
                              "(ex.: ./nllb_tca_checkpoints/checkpoint-200) para continuar "
                              "o treino. Permite mudar outros argumentos (como --eval-steps) "
                              "sem perder o que já foi treinado.")
    parser.add_argument("--freeze-steps", type=int, default=0,
                         help="Congela tudo menos os embeddings por este número de passos "
                              "no início e depois libera para o fine-tuning completo. "
                              "PROBLEMA CONHECIDO: em algumas versões do transformers "
                              "(visto na 5.17.0), congelar via requires_grad causa um "
                              "ValueError de 'decoder_input_ids and decoder_inputs_embeds' "
                              "no forward do M2M100, mesmo sem gradient checkpointing. "
                              "A causa não foi encontrada. Padrão 0 (desligado); use >0 "
                              "só para testar.")
    parser.add_argument("--warmup-steps", type=int, default=300,
                         help="Passos de warmup do LR. Antes era 100 (só ~1,6%% de um treino "
                              "de 6000+ passos), mas o grad_norm ficou muito alto nos "
                              "primeiros passos.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                         help="Label smoothing. 0.1 é o valor usado no Transformer original e no NLLB. "
                              "Suspeito de causar o erro de decoder_input_ids/"
                              "decoder_inputs_embeds no transformers 5.17.0: com >0, "
                              "o Trainer.compute_loss() trata os labels de outro jeito "
                              "antes de chamar o modelo, e isso não dá para reproduzir "
                              "chamando model() direto. Padrão 0 (desligado) até confirmar. "
                              "Use >0 para testar.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "\nERRO: o PyTorch não encontrou uma GPU CUDA nesta máquina "
            "(torch.cuda.is_available() == False).\n"
            "Quase sempre é porque o torch instalado é a versão só para CPU. "
            "Isso é comum no Windows quando se instala com `pip install torch`.\n\n"
            "Veja a versão máxima de CUDA do driver com `nvidia-smi` e "
            "reinstale a versão CUDA correspondente, por exemplo:\n"
            "  pip uninstall torch torchvision torchaudio -y\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
            "(troque cu121 pela versão de CUDA que o driver suporta)\n"
        )
    print(f"GPU CUDA encontrada: {torch.cuda.get_device_name(0)}")

    print("Carregando o tokenizador estendido...")
    tokenizer = AutoTokenizer.from_pretrained(config.EXTENDED_TOKENIZER_DIR)
    new_vocab_size = len(tokenizer)

    print("Carregando o tokenizador original do NLLB para pegar o tamanho do vocabulário...")
    base_tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    old_vocab_size = len(base_tokenizer)
    print(f"Vocabulário original: {old_vocab_size}  ->  estendido: {new_vocab_size}")

    print(f"Carregando o modelo base {config.BASE_MODEL} ...")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.BASE_MODEL)

    print("Redimensionando os embeddings para o vocabulário estendido...")
    # pad_to_multiple_of=8: os tensor cores da arquitetura Ampere são mais
    # rápidos quando a dimensão da camada de saída é múltipla de 8. Custa só
    # algumas linhas sem uso.
    model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=8)
    init_new_embeddings_as_mean(model, old_vocab_size)

    if not args.no_grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False  # necessário junto com o gradient checkpointing
    else:
        print("Gradient checkpointing desligado (--no-grad-checkpointing). "
              "Acompanhe a memória no nvidia-smi; se chegar perto de 10GB, ligue de novo.")

    lang_code_map = build_lang_code_map()
    train_ds = TranslationDataset(config.TRAIN_JSONL, tokenizer, lang_code_map, config.MAX_LEN)
    eval_ds = TranslationDataset(config.TEST_JSONL, tokenizer, lang_code_map, config.MAX_LEN)
    print(f"Exemplos de treino: {len(train_ds)}  |  Exemplos de avaliação: {len(eval_ds)}")

    collate_fn = Collator(pad_id=tokenizer.pad_token_id)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = labels.copy()
        labels[labels == -100] = tokenizer.pad_token_id

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # O chrF++ é mais justo que o BLEU para línguas com muita morfologia e
        # poucos recursos, e não precisa de segmentação de palavras.
        chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)
        return {"chrf++": chrf.score}

    # Os nomes dos argumentos de TrainingArguments/Seq2SeqTrainingArguments
    # mudaram entre versões do transformers (ex.: eval_strategy e o antigo
    # evaluation_strategy; warmup_ratio não existe em algumas instalações).
    # Em vez de fixar os nomes, montamos o dicionário de kwargs e usamos o nome
    # que o __init__ da versão instalada aceita.
    supported = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters)

    def pick_from(supported_names, *name_value_pairs):
        """Recebe pares (nome, valor) em ordem de preferência e devolve um dict
        com o primeiro nome que a versão aceita."""
        for name, value in name_value_pairs:
            if name in supported_names:
                return {name: value}
        return {}

    training_kwargs = dict(
        output_dir=str(config.MODEL_OUTPUT_DIR),
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        bf16=args.bf16,
        gradient_checkpointing=not args.no_grad_checkpointing,
        optim="adamw_bnb_8bit" if args.use_8bit_adam else "adamw_torch",
        logging_steps=25,
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="chrf++",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=config.MAX_LEN,
        # Decodificação gulosa (sem beam search) na avaliação durante o treino.
        # O beam search é bem mais lento e não precisa dessa qualidade só para
        # acompanhar a tendência. A avaliação com beam search fica para depois do
        # treino, com material de estudo real e revisão de um falante.
        generation_num_beams=1,
        label_smoothing_factor=args.label_smoothing,
        report_to=[],  # use ["tensorboard"] para ter log local
        dataloader_num_workers=args.dataloader_workers,
    )
    # Remove o que esta versão não reconhece (raro, mas não custa) e depois
    # adiciona as opções cujo nome muda entre versões.
    training_kwargs = {k: v for k, v in training_kwargs.items() if k in supported}

    effective_batch = args.per_device_batch_size * args.grad_accum
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    total_steps_estimate = max(1, steps_per_epoch * int(args.epochs))
    warmup_ratio_equiv = min(0.5, args.warmup_steps / total_steps_estimate)
    training_kwargs.update(pick_from(supported,
                                      ("warmup_ratio", warmup_ratio_equiv),
                                      ("warmup_steps", args.warmup_steps)))
    training_kwargs.update(pick_from(supported, ("eval_strategy", "steps"), ("evaluation_strategy", "steps")))

    missing = [k for k in ("warmup_ratio", "warmup_steps") if k in training_kwargs]
    if not missing:
        print("OBS: esta versão do transformers não aceita warmup_ratio nem "
              "warmup_steps; o treino vai começar sem warmup.")

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    # O transformers v5 trocou o kwarg tokenizer= do Trainer por processing_class=.
    # Ao contrário dos argumentos acima, não dá para checar isso com confiança
    # pela assinatura do __init__, então tentamos o nome novo e, se der erro,
    # usamos o antigo.
    callbacks = []
    if args.freeze_steps > 0:
        callbacks.append(UnfreezeBackboneCallback(model, args.freeze_steps,
                                                    use_grad_checkpointing=not args.no_grad_checkpointing))

    base_trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    trainer = None
    last_error = None
    for extra_kwargs in ({"processing_class": tokenizer}, {"tokenizer": tokenizer}, {}):
        try:
            trainer = Seq2SeqTrainer(**base_trainer_kwargs, **extra_kwargs)
            break
        except TypeError as e:
            last_error = e
            continue
    if trainer is None:
        raise last_error

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = config.MODEL_OUTPUT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nModelo final e tokenizador salvos em {final_dir}")
    print("\nSobre as métricas: o chrF++ aqui junta todas as direções "
          "(pt->tca, tca->pt, es->tca, tca->es). Quando o treino estabilizar, "
          "separe os resultados por direção e, principalmente, peça revisão "
          "de um falante antes de usar com estudantes.")


if __name__ == "__main__":
    main()
