import * as fs from "fs/promises";
import * as path from "path";
import type { Config } from "./types";

export async function createDirs(config: Config) {
  const DATA_DIR = process.env.DATA_DIR ?? "../dataset";
  const DATASET_SLUG = `omniparser-${config.viewport.width}-${config.viewport.height}`;
  const DATASET_DIR = path.join(DATA_DIR, DATASET_SLUG);
  const WEB_DATASET_DIR = path.join(DATASET_DIR, "/web");
  const IMAGES_DATASET_DIR = path.join(DATASET_DIR, "/images");

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
    console.log("✅ All directories are ready!");
  } catch (error) {
    console.error("❌ Error creating directories:", error);
  }
}
