import { capturePageData, getUrlId, saveDataToJson } from "./lib";

import websitesData from "../websites.json";
import type { Config, WebsitesConfig } from "./types";
import fs from "fs/promises";
import path from "path";

import { createDirs } from "./utils";
import configData from "./config.json";

const config: Config = configData;

const DATA_DIR = process.env.DATA_DIR ?? "../dataset";
const DATASET_SLUG = `omniparser-${config.viewport.width}-${config.viewport.height}`;
const DATASET_DIR = path.join(DATA_DIR, DATASET_SLUG);
const WEB_DATASET_DIR = path.join(DATASET_DIR, "/web");
const IMAGES_DATASET_DIR = path.join(DATASET_DIR, "/images");

const websites: WebsitesConfig = websitesData;

async function main() {
  const outputDir = WEB_DATASET_DIR;
  const imagesDir = IMAGES_DATASET_DIR;
  await createDirs(config);

  for (const category of websites) {
    console.log(`📂 Category: ${category.name}`);

    for (const page of category.pages) {
      console.log(`  └─ 🌐 Domain: ${page.domain}`);

      for (const url of page.urls) {
        const start = performance.now();
        const id = getUrlId(url);

        if (await fs.exists(path.join(imagesDir, `${id}.jpeg`))) {
          console.log(`    └─ ⏭️ Skipping ${id} (already exists)`);
          continue;
        }
        try {
          // Process the page
          await capturePageData({
            url,
            config,
            deleteQueries: page.preprocessing?.deleteQueries,
            cookieQuery: page.preprocessing?.cookieQuery,
            imagesDir,
            outputDir,
          });
        } catch (error) {
          console.log(`  └─ 🟥 error processing page: ${error}`);
          continue;
        }

        const duration = ((performance.now() - start) / 1000).toFixed(2);

        console.log(`    └─ 🔗 ${url} | ✅ done | ⏱️ ${duration}s`);
      }
    }
  }
  process.exit(1);
}

main();
