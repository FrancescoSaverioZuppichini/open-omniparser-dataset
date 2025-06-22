from dataclasses import dataclass

import ipywidgets as widgets
import numpy as np
import requests
import torch
from datasets import load_dataset
from IPython.display import clear_output, display
from PIL import Image, ImageDraw
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.transforms import v2
from transformers import AutoModelForObjectDetection
from transformers.image_transforms import center_to_corners_format


def visualize_detection(
    model: torch.nn.Module,
    image: Image.Image,
    transform,
    image_preprocessor,
    index=0,
    threshold=0.4,
):
    """
    Visualize object detection results for a given test dataset index
    """
    try:

        # Prepare inputs
        inputs = transform(image)
        inputs = inputs.to(model.device)

        # Run inference
        with torch.no_grad():
            outputs = model(pixel_values=inputs.unsqueeze(0))

        target_sizes = torch.tensor([image.size[::-1]])
        result = image_preprocessor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        # Print detection results
        print(f"Image {index} - Detections (threshold={threshold}):")
        print("-" * 50)

        if len(result["scores"]) == 0:
            print("No objects detected!")
        else:
            for score, label, box in zip(
                result["scores"], result["labels"], result["boxes"]
            ):
                box = [round(i, 2) for i in box.tolist()]
                print(
                    f"Detected {model.config.id2label[label.item()]} with confidence "
                    f"{round(score.item(), 3)} at location {box}"
                )

        # Draw bounding boxes
        image_with_boxes = image.copy()
        draw = ImageDraw.Draw(image_with_boxes)

        for score, label, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            box = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = tuple(box)
            draw.rectangle((x, y, x2, y2), outline="red", width=2)
            text_label = model.config.id2label[label.item()]
            draw.text((x, y - 15), f"{text_label} [{score.item():.2f}]", fill="red")

        # Display the image
        return image_with_boxes

    except IndexError:
        print(
            f"Error: Index {index} is out of range. Dataset has {len(dataset['test'])} test images."
        )
