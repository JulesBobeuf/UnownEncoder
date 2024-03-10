# UnownEncoder

UnownEncoder allows you to encrypt latin text to the unown language and vice versa.
Unowns are hieroglyph-like, thin, black ancient Pokémon. There are 28 forms of Unown: one for each of the 26 letters in the Latin alphabet, a question mark and an exclamation mark. 

## Note

The `model_save.pt` file might not be compatible on all operating systems. You may need to re-create one by running `src/train_vit.py`. In this scenario, make sure to not call the current `model_save.pt` in the program.

The initial dataset labels are delayed. Due to this, we must map by hand the classes based on the index.

## Installation

The following guide is based on Linux. On other operating systems like Windows, some commands may slightly change (ex: `python` instead of `python3`)

```shell
git clone https://github.com/JulesBobeuf/UnownEncoder.git
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

## Potential future work :

- Improve GUI
- Improve code quality and readability
- Improve code organization
- Use a .env file for properties
