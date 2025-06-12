import ujson
from datasets import Dataset, Image, Features, Sequence, Value, ClassLabel
import os
from collections import defaultdict


def create_coco_dataset_generator(coco_json_path, images_dir, category_id_to_index):
    with open(coco_json_path) as f:
        coco_data = ujson.load(f)

    img_to_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    for img_info in coco_data["images"]:
        img_id = img_info["id"]
        img_path = os.path.join(images_dir, img_info["file_name"])

        if not os.path.exists(img_path):
            continue

        anns = img_to_anns[img_id]

        objects = []
        for ann in anns:
            objects.append(
                {
                    "id": ann["id"],
                    "area": ann["area"],
                    "bbox": ann["bbox"],
                    "category": category_id_to_index[
                        ann["category_id"]
                    ],  # This maps 1 -> 0
                }
            )

        yield {
            "image_id": img_id,
            "image": img_path,
            "width": img_info["width"],
            "height": img_info["height"],
            "objects": objects,
        }


root = "../datasets/omniparser-1280-800"
with open(os.path.join(root,  "coco.json")) as f:
    coco_data = ujson.load(f)

# Sort categories by ID and create mapping
sorted_categories = sorted(coco_data["categories"], key=lambda x: x["id"])
category_names = [cat["name"] for cat in sorted_categories]

# Create mapping: original COCO ID -> 0-indexed position
category_id_to_index = {cat["id"]: idx for idx, cat in enumerate(sorted_categories)}

print(f"Original category IDs: {[cat['id'] for cat in sorted_categories]}")
print(f"Category mapping: {category_id_to_index}")  # Should show {1: 0}
print(f"Category names: {category_names}")  # Should show ['1']

features = Features(
    {
        "image_id": Value("int64"),
        "image": Image(),
        "width": Value("int32"),
        "height": Value("int32"),
        "objects": Sequence(
            {
                "id": Value("int64"),
                "area": Value("float32"),
                "bbox": Sequence(Value("float32"), length=4),
                "category": ClassLabel(names=category_names),  # Only one class: ['1']
            }
        ),
    }
)

dataset = Dataset.from_generator(
    create_coco_dataset_generator,
    gen_kwargs={
        "coco_json_path": os.path.join(root, "coco.json"),
        "images_dir": os.path.join(root, "images"),
        "category_id_to_index": category_id_to_index,
    },
    features=features,
)

dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
dataset_split.push_to_hub("Francesco/open-omniparser-dataset")
