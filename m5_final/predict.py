"""Milestone 5: predict the diagnosis of one new dermoscopic image.

    python m5_final/predict.py path/to/image.jpg              # 11 diagnoses (the leaderboard model)
    python m5_final/predict.py path/to/image.jpg --model m4   # Benign / Indeterminate / Malignant

From Python:

    from m5_final.predict import predict
    probabilities = predict("path/to/image.jpg")              # {"AKIEC": 0.03, "BCC": 0.61, ...}, adds up to 1

Three rules make a prediction trustworthy (Session 10), and each is one line below:
    the same preprocessing as in validation     val_tf, never the training transform with its random flips
    evaluation mode                              model.eval()
    no gradients                                 torch.no_grad()

Research and teaching only: this is not a medical device.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # the repo root, so that `shared` can be imported

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import resnet18

from shared import config
from shared.deep import val_tf                                       # the ONE inference transform of the project

MODELS = {"11class": config.MODEL_11CLASS, "m4": config.MODEL_M4}


def load_model(path=config.MODEL_11CLASS):
    """Rebuild the network from its file. Returns (model in evaluation mode, list of class names)."""
    saved = torch.load(path, map_location="cpu")                     # {"state_dict": weights, "classes": names, ...}
    model = resnet18()                                               # the same architecture; the weights come from the file
    model.fc = nn.Linear(512, len(saved["classes"]))                 # the same last layer as in training (3 or 11 outputs)
    model.load_state_dict(saved["state_dict"])                       # fails loudly if the file does not fit the network
    model.eval()                                                     # evaluation mode: BatchNorm uses its stored statistics
    return model, saved["classes"]


def predict(image_path, model_path=config.MODEL_11CLASS):
    """Class probabilities of one image: {class name: probability}."""
    model, classes = load_model(model_path)
    image = Image.open(image_path).convert("RGB")                    # always 3 channels, even for a grey or RGBA file
    x = val_tf(image).unsqueeze(0)                                   # validation transform; unsqueeze: a batch of 1 image
    with torch.no_grad():                                            # no gradients: we only predict
        probabilities = torch.softmax(model(x), dim=1)[0]            # scores -> probabilities; [0]: our single image
    return dict(zip(classes, probabilities.tolist()))


if __name__ == "__main__":
    arguments = sys.argv[1:]
    which = "11class"
    if "--model" in arguments:                                       # optional: --model m4 for the 3-class model
        at = arguments.index("--model")
        which = arguments[at + 1]
        arguments = arguments[:at] + arguments[at + 2:]
    if len(arguments) != 1 or which not in MODELS:
        raise SystemExit(__doc__)
    result = predict(arguments[0], MODELS[which])
    for name, p in sorted(result.items(), key=lambda item: -item[1]):      # most likely class first
        print(f"  {name:<14}{p:6.3f}")
    print(f"most likely: {max(result, key=result.get)}")
    print("Research and teaching only. Not a medical device: do not use it to decide about a patient.")
