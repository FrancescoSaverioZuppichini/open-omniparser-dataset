import json
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import random


def generate_random_color():
    """Generate a random bright color for bounding boxes"""
    r = random.randint(100, 255)
    g = random.randint(100, 255)
    b = random.randint(100, 255)
    return (r, g, b)


def visualize_bounding_boxes(base_filename, dataset_dir="../dataset", save_image=True):
    """
    Load the screenshot and JSON data, then draw bounding boxes for each element.

    Args:
        base_filename: Base name without extension (e.g., 'elements_1715213600000')
        dataset_dir: Directory containing the files
    """
    # Construct file paths
    json_path = os.path.join(dataset_dir, "web", f"{base_filename}")

    # Load JSON data
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find JSON file: {json_path}")
        return

    # Get screenshot path from JSON data
    screenshot_path = os.path.join(dataset_dir, data.get("screenshotPath"))
    if not screenshot_path:
        print("Error: No screenshot path found in JSON data")
        return

    # Load the image
    try:
        img = Image.open(screenshot_path)
    except FileNotFoundError:
        print(f"Error: Could not find screenshot file: {screenshot_path}")
        return

    # Create a copy for drawing
    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)

    # Try to load a font, falling back to default if necessary
    try:
        font = ImageFont.truetype("Arial", 12)
    except IOError:
        font = ImageFont.load_default()

    # Draw a bounding box for each element
    elements = data.get("elements", [])
    print(f"Drawing bounding boxes for {len(elements)} elements...")

    for i, element in enumerate(elements):
        box = element.get("boundingBox")
        if not box:
            continue

        # Extract coordinates
        x, y = box.get("x", 0), box.get("y", 0)
        width, height = box.get("width", 0), box.get("height", 0)

        # Generate a random color for this box
        color = generate_random_color()

        # Draw rectangle
        draw.rectangle([(x, y), (x + width, y + height)], outline=color, width=2)

        # Prepare label text
        element_type = element.get("type", "unknown")
        element_text = element.get("text", "")
        label = f"{i+1}: {element_type}"
        if element_text:
            # Truncate long text
            if len(element_text) > 20:
                element_text = element_text[:17] + "..."
            label += f" - '{element_text}'"

        # Draw label background
        text_width, text_height = draw.textbbox((0, 0), label, font=font)[2:]
        draw.rectangle(
            [(x, y - text_height - 4), (x + text_width + 4, y)],
            fill=(255, 255, 255, 200),
        )

        # Draw label text
        draw.text((x + 2, y - text_height - 2), label, fill=color, font=font)

    if save_image:
        output_path = os.path.join(dataset_dir, f"{base_filename}_debug.png")
        img_with_boxes.save(output_path)
        print(f"Debug visualization saved to: {output_path}")

        return output_path
    return img_with_boxes


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Visualize interactive elements from dataset"
    )
    parser.add_argument(
        "file_path",
        help="Annotation file path",
    )

    args = parser.parse_args()

    base_filename = os.path.basename(args.file_path)
    dataset_dir = os.path.dirname(os.path.dirname(args.file_path))
    print(f"filename={base_filename}, dataset_dir={dataset_dir}")
    # Run visualization with the provided arguments
    visualize_bounding_boxes(base_filename, dataset_dir)
