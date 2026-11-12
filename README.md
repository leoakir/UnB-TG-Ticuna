# Tradução português ↔ Ticuna com NLLB-200

Fine-tuning do modelo `facebook/nllb-200-distilled-600M` para traduzir entre
português (`pt`) e Ticuna (`tca`), uma língua indígena isolada com poucos
dados disponíveis. O objetivo é apoiar estudantes Ticuna com material de
estudo traduzido.

O Ticuna não está entre os 200 idiomas do NLLB. Por isso o projeto estende o
tokenizador com subpalavras treinadas em texto Ticuna e cria um código de
idioma novo, `tca_Latn`.

Configurado para treinar em **i7-13700K + RTX 3080 10GB**. O modelo
treinado é convertido para CTranslate2 (int8) e roda só na CPU.

## Estrutura

```
dataset/                 dados paralelos de origem (JSON)
dictionary_cleanup/      scripts de limpeza do dicionário (3 versões)
config.py                caminhos, campos, códigos de idioma e parâmetros
text_utils.py            normalização de texto usada em todo o projeto
01_prepare_data.py       passo 1: monta treino e teste
02_extend_tokenizer.py   passo 2: estende o tokenizador do NLLB
03_train.py              passo 3: fine-tuning
optimize_for_deployment.py  converte o modelo para CTranslate2 int8
build_lang_detector.py   treina o detector pt/tca usado no tradutor
translate_cpu.py         tradutor em CPU (interativo ou uma frase)
eval_*.py                scripts de avaliação
export_training_report.py   exporta o histórico do treino (CSV + gráfico)
diagnose_*.py            scripts de diagnóstico de erros
```

Pastas geradas pelos scripts e fora do git: `work/`, `nllb_tca_tokenizer/`,
`nllb_tca_checkpoints/`, `nllb_tca_ct2/`.

## Dados

Arquivos em `dataset/`, cada um uma lista de registros JSON:

```json
[{"id": "...", "pt": "...", "tca": "..."}, ...]
```

| Arquivo | Conteúdo | Grupo |
|---|---|---|
| `antigotestamento_pt_tca.json` | Antigo Testamento | `bible` |
| `novotestamento_pt_tca.json` | Novo Testamento | `bible` |
| `livrosprogramacao_pt_tca.json` | livros didáticos de programação | `other` |
| `dicionario_es_pt_tca_phrases.json` | frases de exemplo do dicionário (pt, tca, es) | `other` |
| `dicionario_es_pt_tca_word_pairs.json` | pares de palavras do dicionário (pt, tca, es) | `word_pairs` |

Os arquivos do dicionário saem da limpeza em `dictionary_cleanup/cleanup3`.
Se os campos ou arquivos mudarem, ajuste `FILES` e `TRAIN_DIRECTIONS` em
`config.py`. Hoje o treino usa só as direções pt→tca e tca→pt; o espanhol é
carregado, mas não é usado.

## Instalação

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux/macOS

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate sentencepiece sacrebleu bitsandbytes protobuf
```

Use a versão do `torch` que bate com o CUDA do seu driver (veja com
`nvidia-smi`). Para só traduzir ou avaliar com o modelo CT2, basta:

```bash
pip install ctranslate2 transformers sentencepiece sacrebleu
```

## Pipeline de treino

```bash
python 01_prepare_data.py      # gera work/train.jsonl e work/test.jsonl
python 02_extend_tokenizer.py  # estende o tokenizador com subpalavras tca
python 03_train.py             # fine-tuning
```

### 1. Preparação dos dados

- Gera os pares nas direções de `TRAIN_DIRECTIONS`.
- Separa 1000 exemplos de teste: 850 da Bíblia e 150 das frases do dicionário
  (`TEST_SIZE_TOTAL_EXAMPLES` e `TEST_PHRASES_COUNT`). Assim o teste não fica
  só no registro bíblico.
- A separação é feita pelo id do registro, não pela direção. Se um versículo
  vai para o teste, nem pt→tca nem tca→pt dele aparecem no treino. O script
  confere isso no final e para se encontrar vazamento.
- Aplica oversampling só no treino (`OVERSAMPLE_FACTOR`): o grupo `other` é
  repetido 2x; `bible` e `word_pairs` ficam em 1x.
- Grava o texto monolíngue em Ticuna usado no passo 2 (só com dados do treino).

### 2. Tokenizador

Treina um SentencePiece unigram com até 8000 peças (`NEW_SPM_VOCAB_SIZE`) no
texto Ticuna e junta as peças novas ao tokenizador do NLLB sem remover nem
reordenar as existentes. Depois registra o token `tca_Latn`. Se o corpus não
permitir 8000 peças, o script tenta de novo com o máximo possível.

### 3. Treino

Padrões do `03_train.py`:

| Parâmetro | Valor |
|---|---|
| batch por GPU × acumulação | 8 × 4 (batch efetivo 32) |
| épocas | 6 |
| learning rate | 3e-5 |
| warmup | 300 passos |
| precisão | bf16 + gradient checkpointing |
| avaliação e checkpoint | a cada 250 passos (chrF++) |

O melhor checkpoint pelo chrF++ é carregado no final
(`load_best_model_at_end`). Os embeddings das subpalavras novas começam na
média dos embeddings existentes, com um pouco de ruído.

Se faltar memória na GPU:

```bash
python 03_train.py --per-device-batch-size 4 --grad-accum 8
python 03_train.py --use-8bit-adam
```

Para continuar um treino interrompido:

```bash
python 03_train.py --resume-from-checkpoint ./nllb_tca_checkpoints/checkpoint-400
```

Para exportar o histórico do treino:

```bash
python export_training_report.py   # gera training_history.csv e training_history.png
```

## Uso em CPU

```bash
python build_lang_detector.py      # treina o detector pt/tca (lang_detector.json)
python optimize_for_deployment.py  # converte para CT2 int8 em ./nllb_tca_ct2
python optimize_for_deployment.py --benchmark   # compara tamanho e velocidade com o modelo HF
```

A pasta `nllb_tca_ct2/` fica com o modelo, o tokenizador e o detector. É só
copiar essa pasta e o `translate_cpu.py` + `text_utils.py` para outra máquina.

```bash
# modo interativo, detecta o idioma da frase
python translate_cpu.py --model ./nllb_tca_ct2

# uma frase só
python translate_cpu.py --model ./nllb_tca_ct2 --text "Bom dia" --src pt --tgt tca
```

No modo interativo, digite `0` ou `sair` para encerrar.

## Avaliação

| Script | O que avalia |
|---|---|
| `eval_by_source.py` | checkpoint PyTorch no `test.jsonl`, chrF++ por arquivo e direção |
| `eval_by_source2.py` | igual, com opções contra repetição na geração |
| `eval_by_source_full.py` | adiciona BLEU, % de repetição e razão de tamanho |
| `eval_by_source_full_csv.py` | igual ao anterior e grava um CSV com todos os exemplos |
| `eval_by_source_csv_ct2.py` | mesmo teste com o modelo CT2, para comparar com o PyTorch |
| `eval_testset_ct2.py` | teste em TSV (`ticuna_pt_testset.tsv`) com o modelo CT2, com resumo por seção |
| `eval_external_file.py` | arquivo externo no mesmo formato (ex.: `ticuna_salud_parallel_pt_clean.json`) |

Exemplos:

```bash
python eval_by_source_full_csv.py --checkpoint ./nllb_tca_checkpoints/final
python eval_testset_ct2.py --model-dir nllb_tca_ct2 --testset-tsv ticuna_pt_testset.tsv --output-csv eval_testset_results.csv
```

Observações:

- O `ticuna_pt_testset.tsv` vem de um material didático fora do domínio do
  treino (colunas `id, section, paragraph, type, por, tca`).
- No `ticuna_salud_parallel_pt_clean.json` a referência em português foi
  gerada por tradução automática, então a nota mede a concordância com outro
  sistema, não com uma tradução humana.
- As métricas automáticas não substituem a revisão de um falante de Ticuna.

## Diagnóstico

- `diagnose_ct2_vocab.py`: confere se o vocabulário do modelo CT2 bate com o
  tokenizador HF (tamanho e id do `tca_Latn`).
- `diagnose_forward_crash.py`: isola o erro `decoder_input_ids` /
  `decoder_inputs_embeds` chamando o modelo fora do Trainer.

## Licença

O código está sob a licença MIT (ver `LICENSE`).

A licença MIT vale só para o código. Os dados em `dataset/` e
`dictionary_cleanup/` vêm de fontes com direitos próprios (o dicionário
Ticuna–castellano, ©2016 Instituto Lingüístico de Verano, e traduções da
Bíblia) e continuam pertencendo aos seus titulares. Foram incluídos só para
fins acadêmicos e de reprodutibilidade deste trabalho.
