
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torchvision.models import resnet18
import concurrent.futures

from pathlib import Path

ROOT = Path.cwd()

TARGET_PATH = ROOT / "target_model/weights.safetensors"
SUSPECT_DIR = ROOT / "suspect_models"

SUBMISSION_FILE = "submission.csv"


def get_default_device():
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_worker_devices():
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(min(count, 2))]
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return ["mps"]
    return ["cpu"]


def clear_cache(device_str):
    if device_str.startswith("cuda"):
        torch.cuda.empty_cache()


# core
def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

# 3 data trap
def generate_triple_trap(target_model, device, num_samples=512):
    
    print(f"Generating Trap Data ({num_samples} samples each)...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Load a batch of real data
    dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    clean_images, labels = next(iter(loader))
    clean_images, labels = clean_images.to(device), labels.to(device)
    
    target_model.eval()
    
    
    # Blend the first half of the batch with the second half
    half_idx = num_samples // 2
    mixup_images = 0.5 * clean_images[:half_idx] + 0.5 * clean_images[half_idx:]
    with torch.no_grad():
        target_mixup_logits = target_model(mixup_images)
        
    # Catches High-Fidelity boundary copying)
    clean_images.requires_grad = True
    logits = target_model(clean_images)
    loss = F.cross_entropy(logits, labels)
    target_model.zero_grad()
    loss.backward()
    
    epsilon = 0.1
    adv_images = clean_images + epsilon * clean_images.grad.data.sign()
    with torch.no_grad():
        target_adv_logits = target_model(adv_images)
        
    # ood noise
    noise_images = torch.randn_like(clean_images)
    with torch.no_grad():
        target_noise_logits = target_model(noise_images)
        
    # Return the images and the target model's exact output distributions
    return {
        'mixup_img': mixup_images.detach().cpu(),
        'mixup_log': target_mixup_logits.detach().cpu(),
        'adv_img': adv_images.detach().cpu(),
        'adv_log': target_adv_logits.detach().cpu(),
        'noise_img': noise_images.detach().cpu(),
        'noise_log': target_noise_logits.detach().cpu()
    }

# Dual GPU Worker
def evaluate_suspects(suspect_files, traps, device_str):
    device = torch.device(device_str)
    results = []
    
    # Load all traps onto the specific worker's GPU
    mixup_img = traps['mixup_img'].to(device)
    t_mixup_log = traps['mixup_log'].to(device)
    
    adv_img = traps['adv_img'].to(device)
    t_adv_log = traps['adv_log'].to(device)
    
    noise_img = traps['noise_img'].to(device)
    t_noise_log = traps['noise_log'].to(device)
    
    for idx, file_path in enumerate(suspect_files):
        # Parse ID
        try:
            suspect_id = int(file_path.stem)
        except ValueError:
            suspect_id = int(''.join(filter(str.isdigit, file_path.stem)))
            
        # Load Model
        model = make_model()
        model.load_state_dict(load_file(str(file_path), device=device_str), strict=True)
        model.eval().to(device)
        
        with torch.no_grad():
            # Evaluate MixUp Similarity
            s_mixup_log = model(mixup_img)
            mixup_sim = F.cosine_similarity(t_mixup_log.flatten(), s_mixup_log.flatten(), dim=0).item()
            
            # Evaluate Adversarial Similarity
            s_adv_log = model(adv_img)
            adv_sim = F.cosine_similarity(t_adv_log.flatten(), s_adv_log.flatten(), dim=0).item()
            
            # Evaluate OOD Noise Similarity
            s_noise_log = model(noise_img)
            noise_sim = F.cosine_similarity(t_noise_log.flatten(), s_noise_log.flatten(), dim=0).item()
            
        results.append({
            "id": suspect_id,
            "mixup_score": mixup_sim,
            "adv_score": adv_sim,
            "noise_score": noise_sim
        })
        del model
        clear_cache(device_str)
        if (idx + 1) % 30 == 0:
            print(f"[{device_str}] Evaluated {idx + 1}/{len(suspect_files)} models")
    return results

# pipeline
def main():
    
    # Initiate Target and generate Data Traps
    print("1. Extracting Target Model Fingerprints...")
    device = get_default_device()
    print(f"Using device: {device}")
    target_model = make_model()
    target_model.load_state_dict(load_file(str(TARGET_PATH), device=device), strict=True)
    target_model.to(device)
    traps = generate_triple_trap(target_model, device, num_samples=512)
    del target_model
    clear_cache(device)
    suspect_files = sorted(SUSPECT_DIR.glob("*.safetensors"))
    devices = get_worker_devices()
    chunk_size = len(suspect_files) // len(devices) if len(devices) > 0 else len(suspect_files)
    chunks = []
    for idx, dev in enumerate(devices):
        start = idx * chunk_size
        end = len(suspect_files) if idx == len(devices) - 1 else (idx + 1) * chunk_size
        chunks.append((suspect_files[start:end], dev))
    print(f"2. Launching Parallel Evaluation across {len(suspect_files)} models on {', '.join(devices)}...")
    all_results = []
    if len(devices) == 1:
        all_results = evaluate_suspects(suspect_files, traps, devices[0])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(evaluate_suspects, chunk, traps, dev) for chunk, dev in chunks]
            for future in concurrent.futures.as_completed(futures):
                all_results.extend(future.result())
    print("3. Fusing Metrics and Generating Final Submission...")
    df = pd.DataFrame(all_results).sort_values("id")
    scaler = MinMaxScaler()
    norm_mixup = scaler.fit_transform(df[["mixup_score"]])
    norm_adv = scaler.fit_transform(df[["adv_score"]])
    norm_noise = scaler.fit_transform(df[["noise_score"]])
    final_scores = (norm_mixup + norm_adv + norm_noise) / 3.0
    df_submission = pd.DataFrame({
        "id": df["id"],
        "score": final_scores.flatten()
    })
    df_submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"\n✅ Execution Complete. Saved to {SUBMISSION_FILE}")

if __name__ == "__main__":
    main()