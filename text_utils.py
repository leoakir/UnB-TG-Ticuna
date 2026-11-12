"""
Normalização de texto usada em todo o projeto.

Use normalize_text() em todo lugar que lê texto bruto: preparação dos dados,
treino do tokenizador e qualquer script de inferência. Se for usada só em
algumas partes, o modelo vai receber na inferência codificações que nunca
viu no treino, o que é pior do que não normalizar.

A normalização é conservadora: só corrige diferenças de codificação Unicode
que não aparecem a olho nu, mas que mudam a tokenização. Não converte para
minúsculas, não remove diacríticos e não junta vogais repetidas, porque isso
pode ter valor fonêmico na ortografia Ticuna.
"""

import unicodedata

# Variações de apóstrofo/oclusiva glotal encontradas em textos digitalizados
# de fontes diferentes. Antes de confiar nesta lista, vale conferir nos
# arquivos quais desses caracteres aparecem de fato e, se possível, confirmar
# o caractere correto com um falante ou com quem mantém a ortografia.
APOSTROPHE_VARIANTS = [
    "\u2019",  # ' aspas simples direita
    "\u02bc",  # ʼ letra modificadora apóstrofo (recomendada pelo Unicode para a oclusiva glotal)
    "\u2018",  # ' aspas simples esquerda
    "\u0060",  # ` acento grave
    "\u00b4",  # ´ acento agudo (às vezes digitado no lugar do apóstrofo)
]
CANONICAL_APOSTROPHE = "\u0027"  # ' apóstrofo ASCII comum; trocar se o padrão da ortografia for U+02BC


def normalize_text(text: str) -> str:
    if text is None:
        return text

    # 1. Composição canônica (NFC), e não de compatibilidade (NFKC).
    #    - Junta letra + diacrítico no caractere pré-composto quando ele existe
    #      (ex.: "u" + trema -> U+00FC "ü"), o que deixa o texto consistente e
    #      gera menos tokens.
    #    - Quando não existe forma pré-composta (caso de vários diacríticos na
    #      mesma letra), o NFC ainda ordena os diacríticos de forma canônica.
    #      Assim, os mesmos diacríticos digitados em ordem diferente viram o
    #      mesmo texto.
    #    - O NFKC alteraria caracteres de compatibilidade, o que pode mudar o
    #      sentido numa ortografia que não conhecemos bem.
    text = unicodedata.normalize("NFC", text)

    # 2. Troca todas as variações de apóstrofo/oclusiva glotal por um único caractere.
    for variant in APOSTROPHE_VARIANTS:
        text = text.replace(variant, CANONICAL_APOSTROPHE)

    # 3. Limpeza de espaços em branco.
    text = " ".join(text.split())

    return text
