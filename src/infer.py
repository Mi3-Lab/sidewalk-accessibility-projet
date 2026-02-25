import argparse
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import joblib
import json
import numpy as np
from pathlib import Path

CLASS3 = ['no', 'unsure', 'yes']
AIDS = ["Walking cane", "Walker", "Mobility scooter", "Manual wheelchair", "Motorized wheelchair"]

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    return model, processor, device

def load_vision_model(model_name="resnet50"):
    import torchvision.transforms as transforms
    from torchvision.models import resnet50
    import timm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "resnet50":
        model = resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == "vit_base_patch16_224":
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif model_name == "convnext_base":
        model = timm.create_model('convnext_base', pretrained=True)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError("Unsupported model")
    model = model.to(device)
    model.eval()
    return model, transform, device

def predict_image(image_path, models_dir, model, processor, device, model_type="clip"):
    image = Image.open(image_path).convert("RGB")
    if model_type == "clip":
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            feat = outputs.pooler_output / outputs.pooler_output.norm(dim=-1, keepdim=True)
            feat = feat.cpu().numpy()
    else:
        inputs = processor(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            if outputs.dim() == 4:
                feat = outputs.squeeze().flatten(0).cpu().numpy().reshape(1, -1)
            elif outputs.dim() == 3:
                feat = outputs[:, 0, :].cpu().numpy().reshape(1, -1)
            elif outputs.dim() == 2:
                feat = outputs.cpu().numpy().reshape(1, -1)
            else:
                raise ValueError(f"Unexpected output shape: {outputs.shape}")
    
    results = {}
    for aid in AIDS:
        subdir = Path(models_dir) / f"{aid.lower().replace(' ', '_')}_{model_type}"
        if not subdir.exists():
            continue
        scaler = joblib.load(subdir / "scaler.joblib")
        import torch.nn as nn
        probe = nn.Linear(feat.shape[1], 3)
        probe.load_state_dict(torch.load(subdir / "probe.pth"))
        probe.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        probe = probe.to(device)
        
        feat_s = scaler.transform(feat)
        feat_t = torch.tensor(feat_s, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = probe(feat_t)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        probs_full = probs
        pred = np.argmax(probs_full)
        results[aid] = {
            "prediction": CLASS3[pred],
            "probabilities": {CLASS3[i]: float(probs_full[i]) for i in range(3)}
        }
    return results

def main():
    parser = argparse.ArgumentParser(description="Infer accessibility on an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--models_dir", type=str, required=True, help="Directory with trained models")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="Model name")
    parser.add_argument("--model_type", type=str, default="clip", choices=["clip", "resnet50", "vit_base_patch16_224", "convnext_base"], help="Model type")
    args = parser.parse_args()

    if args.model_type == "clip":
        model, processor, device = load_clip_model(args.model_name)
    else:
        model, processor, device = load_vision_model(args.model_name)
    results = predict_image(args.image, args.models_dir, model, processor, device, args.model_type)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()