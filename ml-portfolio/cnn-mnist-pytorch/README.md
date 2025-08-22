# CNN on MNIST (PyTorch)

- Simple ConvNet for MNIST digits.
- Saves best checkpoint to `artifacts/best.pt`.

## Run

```bash
pip install -r ../requirements.txt
python train.py
python infer.py --image_path path/to/digit.png  # optional demo
```
