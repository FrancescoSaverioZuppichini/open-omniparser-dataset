import { capturePageData, getUrlId, saveDataToJson } from "./lib";

import websitesData from "../websites.json";
import type { WebsitesConfig } from "./types";
import fs from "fs/promises";
import path from "path";

import { createDirs } from "./utils";

const DATASET_DIR = process.env.DATASET_DIR ?? "../dataset";
const WEB_DATASET_DIR = path.join(DATASET_DIR, "/web");
const IMAGES_DATASET_DIR = path.join(DATASET_DIR, "/images");

const websites = websitesData as unknown as WebsitesConfig;

// const websites: WebsitesConfig = [
//   {
//     name: "Video Platforms",
//     pages: [
//       {
//         domain: "youtube.com",
//         urls: [
//           "https://www.youtube.com/feed/trending",
//           "https://www.youtube.com/watch?v=VQRLujxTm3c",
//           "https://www.youtube.com/gaming",
//           "https://www.youtube.com/results?search_query=durgasoft",
//           "https://www.youtube.com/music",
//           "https://www.youtube.com/news",
//           "https://www.youtube.com/premium",
//           "https://www.youtube.com/watch?v=XXYlFuWEuKI&list=RDQMgEzdN5RuCXE&start_radio=1",
//         ],
//         preprocessing: {
//           cookieQuery: '[aria-label*="cookie"]',
//         },
//       },
//     ],
//   },
// ];
(async () => {
  const outputDir = WEB_DATASET_DIR;
  const imagesDir = IMAGES_DATASET_DIR;
  await createDirs();
  // const url = "https://commoncrawl.org/get-started";
  // const url = "https://github.com/KwaiVGI/LivePortrait";
  // const url =
  //   "https://github.com/KwaiVGI/LivePortrait/commit/6c4a883a9e67330fdecb0982b0c0611d425c8681";

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
})();
