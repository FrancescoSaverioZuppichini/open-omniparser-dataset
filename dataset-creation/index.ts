import { capturePageData, getUrlId, saveDataToJson } from "./lib";

import StealthPlugin from "puppeteer-extra-plugin-stealth";
import AdblockerPlugin from "puppeteer-extra-plugin-adblocker";
import puppeteer from "puppeteer-extra";
import websitesData from "../websites.json";
import type { WebsitesConfig } from "./types";
import fs from "fs/promises";
import path from "path";

const OMNI_DATASET_DIR = process.env.OMNI_DATASET_DIR ?? "../dataset";
const website = websitesData as unknown as WebsitesConfig;

(async () => {
  const outputDir = OMNI_DATASET_DIR;
  // const url = "https://commoncrawl.org/get-started";
  // const url = "https://github.com/KwaiVGI/LivePortrait";
  // const url =
  //   "https://github.com/KwaiVGI/LivePortrait/commit/6c4a883a9e67330fdecb0982b0c0611d425c8681";

  for (const category of website) {
    console.log(`📂 Category: ${category.name}`);

    for (const page of category.pages) {
      console.log(`  └─ 🌐 Domain: ${page.domain}`);

      for (const url of page.urls) {
        const start = performance.now();
        const id = getUrlId(url);

        if (await fs.exists(path.join(outputDir, `${id}.json`))) {
          console.log(`    └─ ⏭️ Skipping ${id} (already exists)`);
          continue;
        }
        // Process the page
        await capturePageData({
          url,
          deleteQueries: page.preprocessing?.deleteQueries,
          cookieQuery: page.preprocessing?.cookieQuery,
          outputDir,
        });

        const duration = ((performance.now() - start) / 1000).toFixed(2);

        console.log(`    └─ 🔗 ${url} | ✅ done | ⏱️ ${duration}s`);
      }
      process.exit(1);
    }
  }
  process.exit(1);
})();
