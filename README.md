# Pro Flower PyTorch — Flowers102 Classifier + AI Info

Deep learning project for **INFO 6147 — Deep Learning with PyTorch**.

A flower species classifier fine-tuned on the
[Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset
(102 classes), plus a Gradio web app that combines the classifier with an
LLM (OpenAI or local Ollama vision model) to generate descriptions,
habitat info, and care tips for uploaded flower photos.

## Project structure

| File | Description |
|---|---|
| `Project_pytorch_final_24.ipynb` | Final training/evaluation notebook (Scratch Model and ResNet18 fine-tuning, data augmentation, comparison of two training runs) |
| `app.py` | Gradio app: top-5 predictions + AI-generated flower info |
| `best_model_a.pth` | Trained ScratchModel checkpoints |
| `best_model_b.pth` | Trained ResNet18 checkpoints |
| `results/` | Training curves, confusion matrices, class distribution, sample predictions |
| `Pytorch Report 2.pptx` | Presentation slides |

## Setup

```bash
pip install -r requirements.txt
```

The Flowers102 dataset is **not included** in the repo (too large).
Download it and extract to `.data/flowers-102/` before running the notebook.

## Running the app

```bash
python app.py
```

Then open the local Gradio URL it prints (default `http://127.0.0.1:7860`).

### AI info options

- **OpenAI**: `export OPENAI_API_KEY=sk-...` — uses `gpt-4o` (vision).
- **Ollama (free, local)**: install [Ollama](https://ollama.com), then
  `ollama pull gemma3:4b` and run `python app.py`. Vision is enabled
  by default in `app.py` (`OLLAMA_MODEL` / `OLLAMA_VISION` config at the top).

## Results

See `results/` for training curves, confusion matrices, and per-class
accuracy. Model B (in `app.py`) is the checkpoint used for inference.
