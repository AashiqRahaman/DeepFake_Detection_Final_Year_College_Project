import argparse
import os
import sys
import cv2
import numpy as np
import tensorflow as tf

# --- make root (parent of Testing/) importable so we can import model.py ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import model  # gives us ModifiedBranch, MainBranch, Attention


# ------------------------ preprocessing ------------------------
def preprocess_image(path, target_size=(299, 299)):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


# ------------------------ single prediction --------------------
def predict_image(model_obj, path):
    x = preprocess_image(path)
    probs = model_obj.predict(x, verbose=0)[0]   # [p_fake, p_real]
    return {"fake": float(probs[0]), "real": float(probs[1])}


# ------------------------ evaluate folder ----------------------
def evaluate_folder(model_obj, test_dir):
    labels = ["fake", "real"]
    valid_exts = (".png", ".jpg", ".jpeg")
    correct, total = 0, 0

    for label in labels:
        folder = os.path.join(test_dir, label)
        if not os.path.isdir(folder):
            print(f"⚠️ Warning: folder not found → {folder}")
            continue

        for fname in os.listdir(folder):
            if not fname.lower().endswith(valid_exts):
                continue
            fpath = os.path.join(folder, fname)
            try:
                result = predict_image(model_obj, fpath)
            except Exception as e:
                print(f"❌ Error processing {fpath}: {e}")
                continue

            predicted = max(result, key=result.get)
            if predicted == label:
                correct += 1
            total += 1
            print(f"{label}/{fname} → {predicted}  {result}")

    if total == 0:
        print("\nNo valid images found.")
    else:
        print(f"\n✅ Overall Test Accuracy: {correct/total:.2%} ({correct}/{total})")


# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Test Deepfake Detector")
    ap.add_argument("--model", required=True, help="Path to .keras or .h5 model")
    ap.add_argument("--test", required=False, help="Path to test set root (with fake/ and real/)")
    ap.add_argument("--image", required=False, help="Path to a single image to predict")
    args = ap.parse_args()

    print(f"📂 Loading model from {args.model} ...")

    # load with custom objects and safe_mode=False
    custom_objs = {
        "ModifiedBranch": model.ModifiedBranch,
        "MainBranch": model.MainBranch,
        "Attention": model.Attention,
    }
    net = tf.keras.models.load_model(
        args.model,
        custom_objects=custom_objs,
        compile=False,
        safe_mode=False
    )

    if args.image:
        print(f"\n🔍 Predicting single image: {args.image}")
        print(predict_image(net, args.image))
    elif args.test:
        print(f"\n🧪 Evaluating dataset in {args.test}")
        evaluate_folder(net, args.test)
    else:
        print("⚠️ Please provide either --image <path> or --test <dir>.")


if __name__ == "__main__":
    main()
