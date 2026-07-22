# EmotionSense 🌈 (ONNX)

## Final clean structure — this is ALL you need:

```
emotion-app/
├── app.py
├── requirements.txt
├── model.onnx
├── model.onnx.data
└── tokenizer_files/
    └── tokenizer.json
    └── config.json        (optional, only if it has id2label)
```

Everything else from your original project — `.vercel/`, `api/`, `venv/`,
`__pycache__/`, `vercel.json`, `.vercelignore`, `.env.local` — is NOT needed
and should not be pushed to GitHub for this deploy. Keep `.gitignore` and
`.gitattributes` though.

## 1. Add your model files

Drop `model.onnx` + `model.onnx.data` in the root, and your tokenizer files
into `tokenizer_files/`.

## 2. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 3. Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repo.
   - If `model.onnx.data` is over 100MB, use Git LFS:
     ```bash
     git lfs install
     git lfs track "*.onnx" "*.data"
     git add .gitattributes
     git add .
     git commit -m "Add onnx model"
     git push
     ```
2. Go to share.streamlit.io → sign in with GitHub → "New app"
3. Select repo/branch, set main file path to `app.py`
4. Deploy — first load takes ~1 min, then it's cached and fast

## Why ONNX over the original safetensors/transformers version

- No need to install full torch + transformers (huge, slow builds on Streamlit Cloud)
- Faster inference, smaller footprint
- Same trained weights, same accuracy — just a converted runtime format
