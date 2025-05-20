import * as fs from "fs/promises";
import * as path from "path";

const DATASET_DIR = process.env.DATASET_DIR ?? "../dataset";
const WEB_DATASET_DIR = path.join(DATASET_DIR, "/web");
const IMAGES_DATASET_DIR = path.join(DATASET_DIR, "/images");
const COCO_DATASET_DIR = path.join(DATASET_DIR, "/coco");

export async function createDirs() {
  try {
    try {
      await fs.access(DATASET_DIR);
    } catch {
      await fs.mkdir(DATASET_DIR, { recursive: true });
      console.log(`📁 Created directory: ${DATASET_DIR}`);
    }
    try {
      await fs.access(WEB_DATASET_DIR);
    } catch {
      await fs.mkdir(WEB_DATASET_DIR, { recursive: true });
      console.log(`🌐 Created directory: ${WEB_DATASET_DIR}`);
    }
    try {
      await fs.access(IMAGES_DATASET_DIR);
    } catch {
      await fs.mkdir(IMAGES_DATASET_DIR, { recursive: true });
      console.log(`🖼️ Created directory: ${IMAGES_DATASET_DIR}`);
    }
    try {
      await fs.access(COCO_DATASET_DIR);
    } catch {
      await fs.mkdir(COCO_DATASET_DIR, { recursive: true });
      console.log(`🏷️ Created directory: ${COCO_DATASET_DIR}`);
    }
    console.log("✅ All directories are ready!");
  } catch (error) {
    console.error("❌ Error creating directories:", error);
  }
}
