# UnownEncoder

UnownEncoder allows you to encrypt latin text to the unown language and vice versa.
Unown are hieroglyph-like, thin, black ancient Pokémon. There are 28 forms of Unown: one for each of the 26 letters in the Latin alphabet, a question mark and an exclamation mark. 

## Installation

Note : The following guide is based on Linux. On other operating systems like Windows, some commands may slightly change (ex: `python` instead of `python3`)

```shell
git clone
cd UnownEncoder
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the application

```shell
source .venv/bin/activate
python3 src/main.py
```

## Train the ViT model

```shell
source .venv/bin/activate
python3 src/train_vit.py
```

## Evaluate the ViT model

```shell
source .venv/bin/activate
python3 src/eval_vit.py
```
