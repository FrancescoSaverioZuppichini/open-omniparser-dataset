from dataclasses import dataclass

import numpy as np
import torch
from comet_ml import start
from comet_ml.integration.pytorch import log_model
from datasets import load_dataset
from PIL import Image, ImageDraw
from torchvision.transforms import v2
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from dataset import OmniparserDataset
from metrics import MAPEvaluator
from transform import get_transforms

checkpoint = "ustc-community/dfine-medium-obj365"
width, height = 800, 640


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


def log_validation_images_to_comet(
    experiment,
    model,
    validation_dataset,
    image_processor,
    step: int,
    num_images=8,
    threshold=0.4,
):
    """Log validation images with predictions to Comet ML"""
    model.eval()
    device = next(model.parameters()).device

    # Always use the same first N images for consistency
    indices = list(range(0, num_images))

    for i, idx in enumerate(indices):
        try:
            # Get original image
            original_image = validation_dataset.dataset[idx]["image"]

            # Get processed sample
            sample = validation_dataset[idx]
            pixel_values = sample["pixel_values"].unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)

            # Post-process
            target_sizes = torch.tensor([original_image.size[::-1]])
            results = image_processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=target_sizes
            )[0]

            # Draw boxes on original image
            image_with_boxes = original_image.copy()
            draw = ImageDraw.Draw(image_with_boxes)

            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                x1, y1, x2, y2 = box.tolist()
                draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
                label_name = model.config.id2label[label.item()]
                draw.text((x1, y1 - 20), f"{label_name} {score:.2f}", fill="red")

            experiment.log_image(
                image_with_boxes, name=f"val_pred_{idx}_step={step}_th={threshold}"
            )

        except Exception as e:
            print(f"Failed logging image {idx}: {e}")


class CometImageLogger(TrainerCallback):
    def __init__(self, experiment, validation_dataset, image_processor):
        self.experiment = experiment
        self.validation_dataset = validation_dataset
        self.image_processor = image_processor

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if state.global_step % 5 == 0:
            log_validation_images_to_comet(
                self.experiment,
                model,
                self.validation_dataset,
                self.image_processor,
                state.global_step,
                num_images=8,
                threshold=0.4,
            )


if __name__ == "__main__":
    dataset = load_dataset("Francesco/open-omniparser-dataset", cache_dir="data/")

    train_transform, val_transform = get_transforms((width, height))

    image_processor = AutoImageProcessor.from_pretrained(
        checkpoint,
        do_resize=True,
        size={"width": width, "height": height},
        use_fast=True,
    )
    if "validation" not in dataset:
        split = dataset["train"].train_test_split(0.10, seed=1337)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    train_dataset = OmniparserDataset(
        dataset["train"],
        train_transform,
    )
    validation_dataset = OmniparserDataset(
        dataset["validation"],
        val_transform,
    )
    test_dataset = OmniparserDataset(
        dataset["test"],
        val_transform,
    )
    categories = dataset["train"].features["objects"].feature["category"].names
    id2label = {index: x for index, x in enumerate(categories, start=0)}
    label2id = {v: k for k, v in id2label.items()}

    eval_compute_metrics_fn = MAPEvaluator(
        image_processor=image_processor, threshold=0.01, id2label=id2label
    )

    model = AutoModelForObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    experiment = start(
        api_key="I1nY3jSiNB8ob2ETxF1APyRDR",
        project_name="omni-parser",
        workspace="francesco-zuppichini",
    )

    experiment.log_parameters({"image_width": width, "image_height": height})

    model_slug = f"{checkpoint.split('/')[-1]}-{width}x{height}"

    training_args = TrainingArguments(
        output_dir=model_slug,
        fp16=True,
        num_train_epochs=150,
        max_grad_norm=1.0,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=100,
        per_device_train_batch_size=16,
        dataloader_num_workers=8,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=20,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to="comet_ml",
        logging_steps=50,
        logging_strategy="steps",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=image_processor,
        data_collator=collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    trainer.add_callback(
        CometImageLogger(experiment, validation_dataset, image_processor)
    )
    trainer.train()
