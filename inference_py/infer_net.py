import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import argparse
import os
import glob


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


CLASS_NAMES = ["background", "drone"]
INPUT_SIZE = 32
MODEL_PATH = "./model/Net_best.pth"


def load_model(weight_path, device):
    model = Net()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess(img_bgr):
    img = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE))
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(img)
    return tensor.unsqueeze(0)


def infer_single(model, img_bgr, device):
    tensor = preprocess(img_bgr).to(device)
    with torch.no_grad():
        output = model(tensor).squeeze()
        probs = torch.softmax(output, dim=0)
        pred_cls = torch.argmax(probs).item()
        confidence = probs[pred_cls].item()
    return pred_cls, confidence, probs.cpu().numpy()


def infer_image(model, image_path, device):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return
    pred_cls, conf, probs = infer_single(model, img, device)
    print(f"  {os.path.basename(image_path):30s} -> {CLASS_NAMES[pred_cls]} "
          f"(conf={conf:.4f})  [bg={probs[0]:.4f}, drone={probs[1]:.4f}]")
    return pred_cls, conf


def infer_folder(model, folder_path, device):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    files.sort()

    if not files:
        print(f"[WARN] No images found in {folder_path}")
        return

    print(f"Found {len(files)} images in {folder_path}\n")
    stats = {c: 0 for c in CLASS_NAMES}
    for f in files:
        pred_cls, _ = infer_image(model, f, device)
        if pred_cls is not None:
            stats[CLASS_NAMES[pred_cls]] += 1

    print(f"\n--- Statistics ---")
    for cls_name, count in stats.items():
        print(f"  {cls_name}: {count}")


def infer_video(model, video_path, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    cv2.namedWindow("Net Inference", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Net Inference", 640, 480)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        pred_cls, conf, probs = infer_single(model, frame, device)

        label = f"{CLASS_NAMES[pred_cls]} ({conf:.2f})"
        color = (0, 255, 0) if pred_cls == 1 else (0, 0, 255)
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"Frame: {frame_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Net Inference", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames.")


def main():
    parser = argparse.ArgumentParser(description="Net_best.pth Inference (Drone Binary Classifier)")
    parser.add_argument("input", default="/home/verser/Pictures/drone.png", help="Path to image, folder, or video file")
    parser.add_argument("--weights", default=MODEL_PATH, help="Model weights path")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Inference device")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Weights: {args.weights}\n")

    model = load_model(args.weights, device)

    if os.path.isdir(args.input):
        infer_folder(model, args.input, device)
    elif args.input.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
        infer_video(model, args.input, device)
    else:
        infer_image(model, args.input, device)


# python infer_net.py /home/verser/Pictures/drone.png
if __name__ == "__main__":
    main()
