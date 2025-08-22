# ML Portfolio – Starter Projects

A compact set of **runnable machine learning examples** you can push to GitHub.
Each folder is self-contained with a `README.md` and scripts. Designed to be
easy to run locally or in a Codespace/Colab.

## Projects

1. **tabular-classification** – Scikit-learn pipeline on Iris with proper train/test split, model persistence, and evaluation.
2. **cnn-mnist-pytorch** – Simple CNN on MNIST with PyTorch + checkpointing.
3. **nlp-text-classification** – Bag-of-words + logistic regression on a subset of 20 Newsgroups.
4. **fastapi-inference** – REST API (FastAPI) that serves the Iris model with Dockerfile.

## Quickstart

```bash
# (Option A) Create a venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# (Option B) Or use 'uv' / 'pipx' / conda if you prefer

pip install -r requirements.txt
```

Then see each project folder for usage. Push this repo to GitHub:

```bash
git init
git add .
git commit -m "Add ML portfolio starter"
git branch -M main
git remote add origin YOUR_REPO_URL
git push -u origin main
```

---

### Notes
- Code uses **deterministic seeds** where practical.
- No external credentials or paid datasets required.
- MNIST/20NG will download automatically on first run.
