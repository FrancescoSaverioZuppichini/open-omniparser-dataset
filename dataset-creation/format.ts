import type {
  CocoAnnotation,
  CocoCategory,
  CocoDataset,
  CocoImage,
  InteractiveElement,
  PageData,
} from "./types";
import fs from "fs/promises";
import path from "path";

const DATASET_DIR = process.env.DATASET_DIR ?? "../dataset";
const WEB_DATASET_DIR = path.join(DATASET_DIR, "/web");
const IMAGES_DATASET_DIR = path.join(DATASET_DIR, "/images");
const COCO_DATASET_DIR = path.join(DATASET_DIR, "/coco");

async function toCocoDataset(webDatasetDir: string, imagesDir: string) {
  console.log(`📂 Starting COCO dataset conversion from: ${webDatasetDir}`);

  const allFiles = await fs.readdir(webDatasetDir);
  const files = allFiles
    .filter((file) => file.endsWith(".json"))
    .map((file) => path.join(webDatasetDir, file));

  console.log(`  └─ 🗃️ Found ${files.length} JSON files to process`);

  const images: CocoImage[] = [];
  const annotations: CocoAnnotation[] = [];
  const categories: CocoCategory[] = [{ id: 1, name: "interactive_elemenet" }];

  let annotationId = 0;

  for (let i = 0; i < files.length; i++) {
    const filePath = files[i];
    const fileName = path.basename(filePath);
    console.log(
      `  └─ 📄 Processing file ${i + 1}/${files.length}: ${fileName}`
    );

    const start = performance.now();
    const fileContent = await fs.readFile(filePath, "utf-8");
    const pageData: PageData = JSON.parse(fileContent);

    const image: CocoImage = {
      id: i,
      width: pageData.viewport.width,
      height: pageData.viewport.height,
      file_name: path.basename(pageData.screenshotPath),
    };

    images.push(image);
    console.log(
      `    └─ 🖼️ Added image: ${image.file_name} (${image.width}x${image.height})`
    );

    const elements = pageData.elements.filter(
      (el) => {
        const bb = el.boundingBox;
        const area = bb.width * bb.height;
        
        return (
          bb.x >= 0 &&
          bb.y >= 0 &&
          bb.width >= 2 &&
          bb.height >= 2 &&
          bb.x + bb.width <= pageData.page.width &&
          bb.y + bb.height <= pageData.page.height &&
          area >= 4 
        );
      }
    );
    for (let j = 0; j < elements.length; j++) {
      const el = elements[j];
      annotations.push({
        id: annotationId++,
        image_id: i,
        category_id: 1,
        bbox: [
          el.boundingBox.x,
          el.boundingBox.y,
          el.boundingBox.width,
          el.boundingBox.height,
        ],
        area: el.boundingBox.width * el.boundingBox.height,
        iscrowd: 0,
      });
    }

    console.log(`    └─ 📌 Processed ${pageData.elements.length} elements`);

    const end = performance.now();
    console.log(`    └─ ⏱️ Processed in ${(end - start).toFixed(2)}ms`);
  }

  console.log(
    `  └─ 📊 Dataset summary: ${images.length} images, ${annotations.length} annotations`
  );

  const dataset: CocoDataset = { images, annotations, categories };
  return dataset;
}

async function main() {
  console.log(`🚀 Starting COCO dataset conversion process`);
  const dataset = await toCocoDataset(WEB_DATASET_DIR, IMAGES_DATASET_DIR);

  const jsonPath = path.join(COCO_DATASET_DIR, "coco.json");
  console.log(`  └─ 💾 Writing dataset to: ${jsonPath}`);
  await fs.writeFile(jsonPath, JSON.stringify(dataset, null, 2));
  console.log(`  └─ ✅ Conversion complete!`);
}

main();
