import clip

for model_name in clip.available_models():
    model, preprocess = clip.load(model_name, device="cpu")
    print(model_name)
    print('Language Parameters:', sum(p.numel() for p in model.transformer.parameters()))
