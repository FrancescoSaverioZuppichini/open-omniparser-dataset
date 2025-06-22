import torch
from torchvision import tv_tensors
from torch.utils.data import Dataset


class OmniparserDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image_id = sample["image_id"]
        image = sample["image"]
        boxes = sample["objects"]["bbox"]
        categories = sample["objects"]["category"]

        class_labels = torch.tensor(categories, dtype=torch.int64)
        iscrowd = torch.zeros_like(class_labels)
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float16)
        if self.transform:
            boxes = tv_tensors.BoundingBoxes(
                boxes,
                format="XYWH",
                canvas_size=(sample["height"], sample["width"]),
                dtype=torch.float16,
            )
            transformed = self.transform(image, boxes, class_labels)
            image, boxes, class_labels = transformed
        if isinstance(boxes, tv_tensors.BoundingBoxes):
            boxes = boxes.data
        else:
            boxes = torch.tensor(boxes, dtype=torch.float16)
        # fuck hf not telling they need fucking Yolo piece of shit fucking shit format
        boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2

        _, height, width = image.shape
        boxes = boxes / torch.tensor([width, height, width, height])
        area = boxes[:, 2] * boxes[:, 3]
        size = torch.tensor([height, width])

        labels = {
            "class_labels": class_labels,
            "boxes": boxes,
            "area": area,
            "size": size,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([image_id]),
        }

        return {"pixel_values": image, "labels": labels}
