"""
Configuração do fine-tuning do NLLB-600M entre português (pt) / espanhol (es)
e Ticuna (tca).

Se os arquivos JSON usarem outros nomes de campo, ajuste as constantes abaixo.
O resto do pipeline lê tudo daqui, então basta mudar em um lugar.
"""

from pathlib import Path

# -----------------------------------------------------------------------
# Caminhos dos arquivos JSON de origem.
# -----------------------------------------------------------------------
DATA_DIR = Path("./dataset")

FILES = {
    # arquivo -> quais chaves de idioma procurar em cada registro
    "antigotestamento_pt_tca.json":          {"langs": ["pt", "tca"], "group": "bible"},
    "novotestamento_pt_tca.json":            {"langs": ["pt", "tca"], "group": "bible"},
    "livrosprogramacao_pt_tca.json":         {"langs": ["pt", "tca"], "group": "other"},
    "dicionario_es_pt_tca_word_pairs.json": {"langs": ["pt", "tca", "es"], "group": "word_pairs"},
    "dicionario_es_pt_tca_phrases.json":    {"langs": ["pt", "tca", "es"], "group": "other"},
}

# Arquivos que podem fornecer exemplos para o teste. A maior parte do
# teste vem destes dois arquivos.
TEST_ELIGIBLE_FILES = {"antigotestamento_pt_tca.json", "novotestamento_pt_tca.json"}

# Parte extra do teste tirada das frases do dicionário (não da Bíblia),
# para que a avaliação não fique só no registro bíblico. Essas frases saem
# do treino do mesmo jeito que as da Bíblia, sem vazamento. Usamos uma
# quantidade fixa em vez de misturar tudo num sorteio só, senão a parte do
# dicionário poderia ficar pequena demais para dizer alguma coisa.
TEST_PHRASES_FILE = "dicionario_es_pt_tca_phrases.json"
TEST_PHRASES_COUNT = 150  # já incluído em TEST_SIZE_TOTAL_EXAMPLES, não é somado a ele

# -----------------------------------------------------------------------
# Arquivos de saída (gerados por 01_prepare_data.py)
# -----------------------------------------------------------------------
WORK_DIR = Path("./work")
TRAIN_JSONL = WORK_DIR / "train.jsonl"
TEST_JSONL = WORK_DIR / "test.jsonl"
TCA_MONOLINGUAL_TXT = WORK_DIR / "tca_monolingual.txt"  # usado para estender o tokenizador

# -----------------------------------------------------------------------
# Códigos de idioma
# -----------------------------------------------------------------------
# Códigos FLORES-200 que o NLLB já conhece.
NLLB_LANG_CODE = {
    "pt": "por_Latn",
    "es": "spa_Latn",
}
# Código novo criado para o Ticuna. Trocar o sufixo de escrita se o tca
# não for escrito em alfabeto latino.
NEW_LANG_CODE = "tca_Latn"
NEW_LANG_KEY = "tca"

# Direções usadas no treino. Adicione ou remova pares conforme precisar.
# (chave_origem, chave_destino)
TRAIN_DIRECTIONS = [
    ("pt", "tca"),
    ("tca", "pt"),
]

# -----------------------------------------------------------------------
# Oversampling: aplicado só ao treino, nunca ao teste.
# Cada arquivo tem um "grupo" definido acima. Os exemplos de um grupo são
# repetidos esse número de vezes por época. Isso muda o peso de cada fonte
# sem alterar os arquivos nem descartar dados.
#
# word_pairs fica com fator 1 (sem repetição), separado de "other"
# (livrosprogramacao + frases do dicionário). Os pares de palavras só
# ensinam vocabulário solto, sem gramática nem ordem das palavras, e já são
# a maior fonte em volume. Repetir mais faria o modelo aprender a trocar
# palavra por palavra em vez de traduzir frases. "other" mantém o fator
# porque são frases completas fora do registro bíblico, que é justamente o
# que o oversampling quer reforçar.
#
# 01_prepare_data.py mostra as contagens por arquivo e por grupo (antes e
# depois do oversampling). Use esses números para ajustar os fatores. Se
# word_pairs ainda dominar mesmo com fator 1, seria preciso reduzir esse
# grupo (subamostragem), o que o código atual não faz: ele só aceita
# fatores inteiros.
# -----------------------------------------------------------------------
OVERSAMPLE_FACTOR = {
    "bible": 1,
    "other": 2,       # livrosprogramacao + frases do dicionário
    "word_pairs": 1,  # pares palavra-palavra do dicionário, sem repetição
}

# -----------------------------------------------------------------------
# Tamanho do teste (ver a lógica em 01_prepare_data.py)
# -----------------------------------------------------------------------
TEST_SIZE_TOTAL_EXAMPLES = 1000  # contado depois de expandir as direções
RANDOM_SEED = 42

# -----------------------------------------------------------------------
# Modelo / tokenizador
# -----------------------------------------------------------------------
BASE_MODEL = "facebook/nllb-200-distilled-600M"
EXTENDED_TOKENIZER_DIR = Path("./nllb_tca_tokenizer")
NEW_SPM_VOCAB_SIZE = 8000  # subpalavras treinadas no texto tca antes de juntar ao vocabulário

MODEL_OUTPUT_DIR = Path("./nllb_tca_checkpoints")
MAX_LEN = 128
