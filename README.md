<a id="readme-top"></a>

<div align="center">
  <h1 align="center">UnownEncoder</h1>
  <img src="assets\all-letters.png" alt="Unown Encoder Screenshot" width="600">
  <p align="center">
    Encode and decode text into Unown language using Python and AI.
    <br />
    <a href="https://github.com/JulesBobeuf/UnownEncoder">View on GitHub</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#folder-structure">Folder Structure</a></li>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#available-scripts">Available Scripts</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project

UnownEncoder is a Python-based tool that utilizes AI to convert Latin text into the Unown language from Pokémon. It supports both encoding and decoding functionalities, allowing for seamless translation between standard text and Unown symbols.

## Built With

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)  
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)  
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Tkinter](https://img.shields.io/badge/Tkinter-FF6F00?style=for-the-badge)](https://docs.python.org/3/library/tkinter.html)

## Getting Started

### Folder Structure

```markdown
UnownEncoder/
├── 📁 data/             # Dataset for training
├── 📁 src/              # Source code for app & model
│   ├── 📁 app/          # GUI application code
│   ├── 📁 model/        # Model architecture & weights
│   ├── 📁 util/         # Helper functions and utilities
│   ├── 📄 main.py       # GUI application entry point
│   ├── 📄 train_vit.py  # Training script for the model
├── 📄 model_save.pt     # Pre-trained model file
├── 📄 requirements.txt  # Project dependencies
├── 📄 LICENSE           # Project license
└── 📄 README.md         # Project documentation

```

### Prerequisites

Ensure you have the following installed:

```sh
python == 3.9
pip >= 21.0
```

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/JulesBobeuf/UnownEncoder.git
   ```
2. Navigate into the project directory:
   ```sh
   cd UnownEncoder
   ```
3. Create a virtual environment and activate it:
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Available Scripts

```sh
python src/train_vit.py
```
Trains the Vision Transformer model for encoding/decoding tasks.

## Usage

Launch the application by running the main GUI file:

```sh
python src/main.py
```

This will open the Tkinter GUI, where you can:

* Encode text into Unown language
* Decode Unown language back to standard text

## License

Licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

Jules Bobeuf  
[LinkedIn](https://www.linkedin.com/in/bobeuf-jules/)  
bobeuf.jules@gmail.com
