import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from fiftyone.brain.visualization import visualize
from torch.utils.data import Dataset
import h5py
import os
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torchvision import transforms
import matplotlib
from torchvision.transforms import InterpolationMode
from PIL import Image
from collections import defaultdict
matplotlib.use('TkAgg')

def collect_sample_paths(top_dir, max_frames_per_subset=None):
    sample_paths = []

    for subset_folder in os.listdir(top_dir):
        outer_path = os.path.join(top_dir, subset_folder)
        nested_path = outer_path
        images_path = os.path.join(nested_path, 'images')
        if not os.path.isdir(images_path):
            print(f"[!] Skipping: {images_path} not found.")
            continue

        for cam_folder in os.listdir(images_path):
            if not cam_folder.endswith("_final_hdf5"):
                continue

            final_path = os.path.join(images_path, cam_folder)
            geom_path = os.path.join(images_path, cam_folder.replace("_final_hdf5", "_geometry_hdf5"))

            if not os.path.isdir(final_path) or not os.path.isdir(geom_path):
                continue

            print(f"📂 Processing {subset_folder}/{cam_folder}...")
            color_files = sorted([f for f in os.listdir(final_path) if f.endswith(".color.hdf5")])
            print(f"  → Found {len(color_files)} .color.hdf5 files")

            selected_files = color_files if max_frames_per_subset is None else color_files[:max_frames_per_subset]

            valid_pairs = 0
            for fname in selected_files:
                frame_id = fname.replace(".color.hdf5", "")
                color_fp = os.path.join(final_path, fname)
                semantic_fp = os.path.join(geom_path, f"{frame_id}.semantic.hdf5")

                if os.path.exists(semantic_fp):
                    sample_paths.append((color_fp, semantic_fp))
                    valid_pairs += 1

            print(f"  ✅ Matched {valid_pairs} image-mask pairs")

    print(f"\n📦 Total matched samples: {len(sample_paths)}")
    return sample_paths


class HypersimSegmentationDataset(Dataset):
    def __init__(self, sample_paths, transform=transforms.ToTensor(), target_transform=None):
        self.sample_paths = sample_paths
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        color_path, semantic_path = self.sample_paths[idx]

        with h5py.File(color_path, 'r') as f_color:
            rgb = np.array(f_color['dataset'])  # shape (H, W, 3)

        with h5py.File(semantic_path, 'r') as f_sem:
            mask = np.array(f_sem['dataset'])  # shape (H, W)

        # Normalize RGB from [0, 1] float → uint8 image
        rgb_disp = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        if self.transform:
            rgb_disp = self.transform(rgb_disp)
        if self.target_transform:
            mask = self.target_transform(mask)

        return rgb_disp, mask

base_dir = r"C:\Users\dania\Deep_learning\scenes"
sample_paths = collect_sample_paths(base_dir)

dataset = HypersimSegmentationDataset(sample_paths)
rgb_img, mask = dataset[999]
mask.shape
rgb_img.shape
plt.imshow(rgb_img.permute(1, 2, 0))  # [C, H, W] → [H, W, C]
plt.axis('off')
plt.show()

#______________________Data_Preparation_Class_For_CLipSeg_______________________

id_to_prompt = {
    1: "a wall",
    2: "the floor",
    3: "a cabinet",
    4: "a bed",
    5: "a chair",
    6: "a sofa",
    7: "a table",
    8: "a door",
    9: "a window",
    10: "a bookshelf",
    11: "a picture",
    12: "a counter",
    13: "blinds",
    14: "a desk",
    15: "shelves",
    16: "a curtain",
    17: "a dresser",
    18: "a pillow",
    19: "a mirror",
    20: "a floormat",
    21: "clothes",
    22: "the ceiling",
    23: "books",
    24: "a refrigerator",
    25: "a television",
    26: "paper",
    27: "a towel",
    28: "a shower curtain",
    29: "a box",
    30: "a whiteboard",
    31: "a person",
    32: "a nightstand",
    33: "a toilet",
    34: "a sink",
    35: "a lamp",
    36: "a bathtub",
    37: "a bag",
    38: "another structure",
    39: "another piece of furniture",
    40: "another object"
}
id_to_prompt.keys()
import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ClipSegHypersimDataset(Dataset):
    def __init__(self, sample_paths, class_ids=None, transform=None, target_transform=None):
        self.sample_entries = []
        self.transform = transform
        self.target_transform = target_transform

        # Store class ID to prompt mapping
        self.class_id_to_prompt = class_ids or {}

        for color_path, semantic_path in sample_paths:
            with h5py.File(semantic_path, 'r') as f_sem:
                mask = np.array(f_sem['dataset'])

            unique_ids = np.unique(mask)
            valid_ids = [cid for cid in unique_ids if cid in self.class_id_to_prompt]

            for class_id in valid_ids:
                self.sample_entries.append((color_path, semantic_path, class_id))

    def __len__(self):
        return len(self.sample_entries)

    def __getitem__(self, idx):
        color_path, semantic_path, class_id = self.sample_entries[idx]

        with h5py.File(color_path, 'r') as f_color:
            rgb = np.array(f_color['dataset'])

        with h5py.File(semantic_path, 'r') as f_sem:
            mask = np.array(f_sem['dataset'])

        # Normalize to uint8 RGB image
        rgb_disp = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        binary_mask = (mask == class_id).astype(np.uint8) * 255

        img = Image.fromarray(rgb_disp)
        mask_img = Image.fromarray(binary_mask)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            mask_img = self.target_transform(mask_img)

        prompt = self.class_id_to_prompt[class_id]
        return img, mask_img, prompt



#Checking if prompts,Image and Mask match
def inspect_sample(dataset, idx):
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    img, mask, prompt = dataset[idx]

    # Convert to NumPy if needed
    img_np = img.permute(1, 2, 0).numpy()  # [C, H, W] → [H, W, C]
    mask_np = mask.numpy().squeeze() if isinstance(mask, torch.Tensor) else np.array(mask)

    # Sanity checks
    assert img_np.shape[:2] == mask_np.shape[:2], "Image and mask size mismatch!"
    assert np.isin(np.unique(mask_np), [0, 255]).all(), "Mask must be binary (0, 255)!"

    # Show
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title(f"Image\nPrompt: {prompt}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap='gray')
    plt.title("Binary Mask")
    plt.axis('off')

    plt.show()

    print(f"Prompt: {prompt}")
    print(f"Image shape: {img_np.shape}")
    print(f"Unique mask values: {np.unique(mask_np)}")


#__________________ClipSeg_model_________________

from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from torchvision.transforms.functional import to_pil_image

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd16")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd16")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

#Picking only 15 classes
_15id_to_prompt = {
    1: "a wall",
    2: "the floor",
    16: "a curtain",
    4: "a bed",
    5: "a chair",
    6: "a sofa",
    7: "a table",
    8: "a door",
    9: "a window",
    10: "a bookshelf",
    11: "a picture",
    18: "a pillow",
    13: "blinds",
    14: "a desk",
    15: "shelves"}

_3id_to_prompt = {
    2: "the floor",
    4: "a bed",
    5: "a chair",
    7: "a table"}

def limit_dataset_per_class(sample_paths, class_ids, limit=50):
    """
    Returns a new sample_paths list limited to `limit` samples per class.
    """
    class_buckets = defaultdict(list)

    for color_path, semantic_path in sample_paths:
        with h5py.File(semantic_path, 'r') as f_sem:
            mask = np.array(f_sem['dataset'])
        unique_ids = np.unique(mask)

        for cid in unique_ids:
            if cid in class_ids and len(class_buckets[cid]) < limit:
                class_buckets[cid].append((color_path, semantic_path))
                break  # take only one matching class per image (optional)

    limited_paths = []
    for samples in class_buckets.values():
        limited_paths.extend(samples)

    return limited_paths

selected_class_ids = list(_3id_to_prompt.keys())  # 15 classes
#Limit number of smaples per class, I got 30
sample_paths_limited = limit_dataset_per_class(sample_paths, selected_class_ids, limit=30)


transform = transforms.Compose([
    transforms.Resize((352, 352), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor()
])

target_transform = transforms.Compose([
    transforms.Resize((352, 352), interpolation=InterpolationMode.NEAREST),
    transforms.PILToTensor()
])
filtered_dataset = ClipSegHypersimDataset(
    sample_paths=sample_paths_limited,
    class_ids=_3id_to_prompt,
    transform=transform,
    target_transform=target_transform
)

#Inspect some sample if their Prompt,Mask and Image match
img,mask,prompt = filtered_dataset[0]
mask.shape
for i in [53]:
    inspect_sample(filtered_dataset, i)


path = r"C:\Users\dania\Deep_learning\dl_team_project\Plots"

def visualize(model, prompt, img, threshold=None, save_dir=path, base_filename="output"):
    # Convert image to PIL
    img_pil = to_pil_image(img)

    # Prepare input
    inputs = processor(text=[prompt], images=[img_pil], return_tensors="pt").to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        pred_mask = outputs.logits[0].sigmoid().cpu().numpy()

        # Threshold
        if threshold is None:
            binary_mask = (pred_mask * 255).astype(np.uint8)
        else:
            binary_mask = (pred_mask > threshold).astype(np.uint8) * 255

    # Convert image tensor to NumPy
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    # === Show with bright green overlay ===
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_np)
    plt.imshow(binary_mask, cmap='Greens', alpha=0.6)  # Higher alpha
    plt.title(f"Bright Green Mask: {prompt}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # === Save image and overlay ===
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # Save original image
        image_path = os.path.join(save_dir, f"{base_filename}_image.png")
        Image.fromarray(img_np).save(image_path)
        print(f"[✓] Image saved to: {image_path}")

        # Create bright green overlay
        green_mask = np.zeros_like(img_np)
        green_mask[..., 1] = binary_mask  # green channel

        # Brighten it by increasing the blend ratio
        overlay = np.clip(img_np * 0.4 + green_mask * 0.6, 0, 255).astype(np.uint8)

        mask_path = os.path.join(save_dir, f"{base_filename}_mask_green.png")
        Image.fromarray(overlay).save(mask_path)
        print(f"[✓] Bright green mask saved to: {mask_path}")


img, mask, prompt = filtered_dataset[50]
#Input to one of the models and check the output
visualize(model,"Chair and Table",img,threshold = 0.3)

#______________________Calculating_Metrics________________________
model_name = "CIDAS/clipseg-rd64-refined"
processor = CLIPSegProcessor.from_pretrained(model_name)
model = CLIPSegForImageSegmentation.from_pretrained(model_name).to(device)
model.eval()
def compute_segmentation_metrics(pred_mask, gt_mask, threshold=0.5):

    # Binarize predictions and GT
    pred_bin = (pred_mask > threshold).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)

    TP = np.logical_and(pred_bin == 1, gt_bin == 1).sum()
    TN = np.logical_and(pred_bin == 0, gt_bin == 0).sum()
    FP = np.logical_and(pred_bin == 1, gt_bin == 0).sum()
    FN = np.logical_and(pred_bin == 0, gt_bin == 1).sum()

    # Avoid division by zero
    epsilon = 1e-7

    accuracy = (TP + TN) / (TP + TN + FP + FN + epsilon)
    precision = TP / (TP + FP + epsilon)
    sensitivity = TP / (TP + FN + epsilon)  # Recall
    specificity = TN / (TN + FP + epsilon)
    f1_score = (2 * TP) / (2 * TP + FP + FN + epsilon)
    iou = TP / (TP + FP + FN + epsilon)

    return {
        'F1': f1_score,
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'IoU': iou
    }

def evaluate_clipseg_one_model(model, processor, loader, threshold=0.5):
    device = next(model.parameters()).device
    class_metrics = defaultdict(list)

    for batch in tqdm(loader):
        img_tensor, gt_mask_tensor, prompt = batch
        prompt = list(prompt)  # unpack prompt from tuple

        # Convert tensor to PIL image
        img_pil = transforms.ToPILImage()(img_tensor[0])
        gt_mask = gt_mask_tensor[0].squeeze().numpy()

        # Prepare input
        inputs = processor(text=prompt, images=[img_pil], return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred_mask = outputs.logits[0].sigmoid().cpu().numpy()

        # Compute metrics
        metrics = compute_segmentation_metrics(pred_mask, gt_mask, threshold)
        class_metrics[prompt[0]].append(metrics)

    # Average per-class
    avg_metrics = {}
    for prompt, metric_list in class_metrics.items():
        keys = metric_list[0].keys()
        avg = {k: sum(d[k] for d in metric_list) / len(metric_list) for k in keys}
        avg_metrics[prompt] = avg

    return avg_metrics


loader = DataLoader(
    filtered_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)
metrics_per_class = evaluate_clipseg_one_model(model, processor, loader)

classes = list(metrics_per_class.keys())
f1_scores = [metrics_per_class[c]['F1'] for c in classes]
accuracies = [metrics_per_class[c]['Accuracy'] for c in classes]
sensitivities = [metrics_per_class[c]['Sensitivity'] for c in classes]
specificities = [metrics_per_class[c]['Specificity'] for c in classes]

x = np.arange(len(classes))
width = 0.2  # bar width

plt.figure(figsize=(16, 6))
plt.bar(x - 1.5*width, f1_scores, width, label='F1 Score')
plt.bar(x + 0.5*width, sensitivities, width, label='Sensitivity')
plt.bar(x + 1.5*width, specificities, width, label='Specificity')

plt.xticks(x, classes, rotation=45, ha='right')
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.title("Per-Class Segmentation Metrics")
plt.legend()
plt.tight_layout()
plt.show()



