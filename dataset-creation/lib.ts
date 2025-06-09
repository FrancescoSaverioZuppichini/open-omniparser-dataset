// DISCLAIMER this code is shit
import { Browser, Page, ElementHandle } from "puppeteer";
import puppeteer from "puppeteer-extra";
import fs from "fs/promises";
import path from "path";
import type { Config, InteractiveElement, PageData } from "./types";
import StealthPlugin from "puppeteer-extra-plugin-stealth";
import AdblockerPlugin from "puppeteer-extra-plugin-adblocker";

async function acceptCookieConsent(
  page: Page,
  timeout: number = 1000
): Promise<boolean> {
  console.log(`  └─ 🍪 Checking for common cookie consent...`);

  // Common cookie consent button selectors
  const selectors: string[] = [
    // Facebook
    'button[data-cookiebanner="accept_button"]',
    'button[data-testid="cookie-policy-manage-dialog-accept-button"]',
    'button[data-testid="cookie-policy-dialog-accept-button"]',

    // Google services
    "button#L2AGLb",
    "button.tHlp8d",

    // Common frameworks/implementations
    "button#onetrust-accept-btn-handler",
    "button.accept-cookies-button",
    "button.consent-accept",
    "button.cookie-accept",
    "button.accept-all-cookies",
    ".cookie-banner .accept-button",
    "#accept-cookies",
    "#cookie-notice .accept",
    ".cc-accept",
    ".gdpr-consent-button",
    ".cookie-consent-accept",
    ".cookie-banner__accept",
    '[aria-label="Accept cookies"]',
    '[aria-label="Accept all cookies"]',
    ".js-accept-cookies",
    ".js-accept-all-cookies",
    ".cookie-consent__accept-button",
    'div[class*="cookie"] button:first-child',
    'div[id*="cookie"] button:first-child',
    ".CybotCookiebotDialogBodyButton",
    ".eu-cookie-compliance-default-button",
    '[aria-label="Allow all cookies"]',
    '[aria-label*="cookie"]',
  ];

  try {
    // Fast check if any selectors exist without waiting
    const buttonSelector = await page.evaluate((selectors) => {
      for (const selector of selectors) {
        const element = document.querySelector(selector);
        if (element && element.isConnected) {
          return selector;
        }
      }
      return null;
    }, selectors);

    if (buttonSelector) {
      await page.click(buttonSelector);
      await page.evaluate(
        () => new Promise((resolve) => setTimeout(resolve, 500))
      );
      console.log(
        `    └─ ✅ Accepted cookies (${buttonSelector.slice(0, 30)}${
          buttonSelector.length > 30 ? "..." : ""
        })`
      );
      return true;
    }

    console.log(`    └─ ℹ️ No cookie banner detected`);
    return false;
  } catch (error) {
    console.log(`    └─ ⚠️ Error handling cookies: ${error}`);
    return false;
  }
}
async function captureInteractiveElements(
  page: Page
): Promise<InteractiveElement[]> {
  // Execute script in the page context to find interactive elements
  return await page.evaluate(() => {
    const interactiveElements: InteractiveElement[] = [];
    // Common interactive element selectors
    const selectors = [
      "button",
      "a",
      // "p",
      // "span",
      "input",
      "select",
      "textarea",
      "details",
      '[role="button"]',
      '[role="link"]',
      '[role="checkbox"]',
      '[role="menuitem"]',
      '[role="tab"]',
      '[role="combobox"]',
      '[role="menu"]',
      '[role="menubar"]',
      '[role="radio"]',
      '[role="switch"]',
      '[role="slider"]',
      '[tabindex]:not([tabindex="-1"])',
      "[onclick]", // Elements with onclick attribute
      "[onmousedown]", // Other mouse event attributes
      "[onmouseup]",
      "[onmouseover]",
      "[onmouseenter]",
      "[onmouseleave]",
      ".btn",
      ".button", // Common CSS classes for buttons
      "*[class*='btn']", // Elements with 'btn' in their class
      "*[class*='click']", // Elements with 'click' in their class
      "*[style*='cursor: pointer']", // Elements with inline pointer cursor
      "[data-toggle]",
    ];
    // Find all elements matching our selectors
    let elements = Array.from(document.querySelectorAll(selectors.join(",")));

    // Additionally find elements with computed style cursor:pointer
    const allElements = Array.from(document.querySelectorAll("*"));
    const pointerElements = allElements.filter((el) => {
      const computedStyle = window.getComputedStyle(el);
      return computedStyle.cursor === "pointer";
    });

    // Combine both sets of elements (removing duplicates)
    elements = [...new Set([...elements, ...pointerElements])];

    // Create a map to track which elements will be included in the final list
    const elementInclusionMap = new Map<Element, boolean>();
    elements.forEach((el) => elementInclusionMap.set(el, true));

    // Create a map for enhanced text descriptions
    const enhancedTextMap = new Map<Element, string>();
    // Map to store descriptions from priority children
    const childDescriptionsMap = new Map<Element, string[]>();

    // Priority tags whose text content should be propagated to parents
    const priorityTags = ["input", "button", "textarea", "select"];

    // First pass: Mark elements that should be suppressed and collect text from priority elements
    elements.forEach((element) => {
      // Initialize with current element's text content
      let tagName = element.tagName.toLowerCase();
      let elementText = "";

      // Extract text based on element type
      if (tagName === "input") {
        const inputEl = element as HTMLInputElement;
        if (inputEl.placeholder) {
          elementText = inputEl.placeholder;
        } else if (inputEl.value && inputEl.type !== "password") {
          elementText = inputEl.value;
        } else if (inputEl.type) {
          elementText = `${inputEl.type} input`;
        }
      } else {
        elementText = (element.textContent || "").trim();
      }

      enhancedTextMap.set(element, elementText);

      // Check if this element should be suppressed (child of another interactive element)
      let parent: Element | null = element.parentElement;
      while (parent) {
        if (elementInclusionMap.has(parent)) {
          // This element has an interactive parent, so it should be suppressed
          elementInclusionMap.set(element, false);

          // If this is a priority element and has meaningful text, propagate to parent
          if (priorityTags.includes(tagName) && elementText) {
            if (!childDescriptionsMap.has(parent)) {
              childDescriptionsMap.set(parent, []);
            }
            childDescriptionsMap.get(parent)!.push(elementText);
          }
          break;
        }
        parent = parent.parentElement;
      }
    });

    // Second pass: Enhance descriptions with priority children's text
    elements.forEach((element) => {
      if (childDescriptionsMap.has(element)) {
        const childTexts = childDescriptionsMap.get(element)!;
        let currentText = enhancedTextMap.get(element) || "";

        if (childTexts.length > 0) {
          const childDescription = childTexts.join(" | ");
          if (currentText) {
            enhancedTextMap.set(
              element,
              `${currentText} (${childDescription})`
            );
          } else {
            enhancedTextMap.set(element, childDescription);
          }
        }
      }
    });

    // Filter elements that should be included
    elements = elements.filter((el) => elementInclusionMap.get(el));

    // Now process the filtered elements to create InteractiveElement objects
    elements.forEach((element) => {
      const rect = element.getBoundingClientRect();

      // Skip zero-size elements
      if (rect.width === 0 || rect.height === 0) return;

      // Get computed style to check visibility
      const style = window.getComputedStyle(element);
      if (
        style.display === "none" ||
        style.visibility === "hidden" ||
        style.opacity === "0"
      )
        return;

      // Get the enhanced text for this element
      let text = enhancedTextMap.get(element) || "";

      // Normalize whitespace
      text = text.replace(/\s+/g, " ");

      // Get attributes
      const attributes: Record<string, string> = {};
      Array.from(element.attributes).forEach((attr) => {
        attributes[attr.name] = attr.value;
      });

      // Generate an XPath for the element
      function getXPath(element: Element): string {
        if (element.id) {
          return `//*[@id="${element.id}"]`;
        }

        if (element === document.body) {
          return "/html/body";
        }

        if (!element.parentElement) {
          return "";
        }

        const sameTagSiblings = Array.from(
          element.parentElement.children
        ).filter((sibling) => sibling.tagName === element.tagName);

        const idx = sameTagSiblings.indexOf(element) + 1;

        return `${getXPath(
          element.parentElement
        )}/${element.tagName.toLowerCase()}[${idx}]`;
      }

      interactiveElements.push({
        type: element.tagName.toLowerCase(),
        text,
        boundingBox: {
          x: rect.x,
          y: rect.y + window.scrollY,
          width: rect.width,
          height: rect.height,
        },
        attributes,
        isVisible: true,
        xpath: getXPath(element),
      });
    });

    return interactiveElements;
  });
}

export async function capturePageData({
  url,
  config,
  deleteQueries,
  outputDir,
  imagesDir,
  cookieQuery,
}: {
  url: string;
  config: Config;
  deleteQueries?: string[];
  cookieQuery?: string;
  outputDir: string;
  imagesDir: string;
}): Promise<PageData[]> {
  const id = getUrlId(url);
  console.log(`🔄 Processing: ${url}`);

  puppeteer.use(StealthPlugin());
  puppeteer.use(AdblockerPlugin({ blockTrackers: true }));

  // Launch the browser
  const browser = await puppeteer.launch({
    headless: true,
  });

  try {
    const page: Page = await browser.newPage();

    await page.setUserAgent(
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    );

    // Set viewport with fixed height
    const viewport = {
      width: config.viewport.width,
      height: config.viewport.height * 2,
    };
    await page.setViewport(viewport);

    // Navigation with timing
    const navStart = performance.now();
    await page.goto(url, {
      waitUntil: "networkidle2",
      timeout: 60000,
    });
    console.log(
      `  └─ 🌐 Loaded page (${((performance.now() - navStart) / 1000).toFixed(
        2
      )}s)`
    );
    await acceptCookieConsent(page);

    // Process page modifications
    if (deleteQueries) {
      let totalDeleteQueries = 0;
      await page.evaluate((deleteQueries) => {
        // Process each selector independently
        for (const selector of deleteQueries) {
          const elements = document.querySelectorAll(selector);
          elements.forEach((element) => {
            element.remove();
            totalDeleteQueries++;
          });
        }
      }, deleteQueries);

      console.log(
        `  └─ 🗑️ Removed ${totalDeleteQueries} element using deleteQueries`
      );
    }

    if (cookieQuery) {
      try {
        await page.click(cookieQuery);
        await page.evaluate(
          () => new Promise((resolve) => setTimeout(resolve, 500))
        );
        console.log(`  └─ 🍪 Handled cookie consent`);
      } catch (error) {
        console.log(
          `  └─ ⚠️ Cookie consent not found for selector: ${cookieQuery}`
        );
      }
    }

    // Get page dimensions
    const dimensions = await page.evaluate(() => ({
      pageHeight: Math.max(
        document.body.scrollHeight,
        document.documentElement.scrollHeight,
        document.body.offsetHeight,
        document.documentElement.offsetHeight
      ),
      pageWidth: Math.max(
        document.body.scrollWidth,
        document.documentElement.scrollWidth,
        document.body.offsetWidth,
        document.documentElement.offsetWidth
      ),
    }));

    // Calculate number of segments needed
    const segmentHeight = config.viewport.height;
    const numSegments = Math.ceil(dimensions.pageHeight / segmentHeight);
    console.log(
      `  └─ 📏 Page height: ${dimensions.pageHeight}px, splitting into ${numSegments} segments`
    );

    // Array to store PageData objects for each segment
    const pageDataResults: PageData[] = [];
    const baseId = getUrlId(url);

    // We store a full screenshot for context
    await page.screenshot({
      path: path.join(imagesDir, `${id}.jpeg`),
      fullPage: true,
      type: "jpeg",
      quality: config.screenshot.quality,
    });

    // Capture each segment, max 12 segments per page
    for (let i = 0; i < Math.min(numSegments, 12); i++) {
      const scrollPosition = i * segmentHeight;
      const segmentIndex = i + 1;

      // Scroll to position
      await page.evaluate((scrollY) => {
        window.scrollTo(0, scrollY);
      }, scrollPosition);

      await page.evaluate(
        () => new Promise((resolve) => setTimeout(resolve, 1000))
      );

      // Take screenshot of current viewport
      const screenshotFilename = `${id}_segment_${segmentIndex}.jpeg`;
      const segmentPath = path.join(imagesDir, screenshotFilename);

      await page.screenshot({
        path: segmentPath,
        fullPage: false,
        type: "jpeg",
        clip: {
          x: 0,
          y: config.viewport.height * i,
          height: config.viewport.height,
          width: config.viewport.width,
        },
        quality: config.screenshot.quality,
      });

      console.log(
        `  └─ 📸 Segment ${segmentIndex}/${numSegments} screenshot saved`
      );

      // Capture interactive elements in current viewport
      const elementsStart = performance.now();
      const visibleElements = await captureInteractiveElements(page);

      // Process elements for this segment
      let segmentElements = visibleElements.map((element) => ({
        ...element,
        boundingBox: {
          ...element.boundingBox,
          y: element.boundingBox.y - scrollPosition,
        },
      }));

      segmentElements = segmentElements.filter(
        ({ boundingBox }) => boundingBox.y <= 800
      );

      console.log(
        `  └─ 🔍 Found ${
          segmentElements.length
        } elements in segment ${segmentIndex} (${(
          (performance.now() - elementsStart) /
          1000
        ).toFixed(2)}s)`
      );

      // Create segment-specific ID
      const segmentId = `${baseId}_segment_${segmentIndex}`;

      // Create PageData object for this segment
      const segmentPageData: PageData = {
        url,
        id: segmentId,
        viewport: { ...viewport, height: viewport.height / 2 },
        page: { height: dimensions.pageHeight, width: dimensions.pageWidth },
        screenshotPath: `images/${screenshotFilename}`,
        elements: segmentElements,
      };

      // Save this segment's data to JSON
      const jsonPath = path.join(outputDir, `${segmentId}.json`);
      await fs.writeFile(jsonPath, JSON.stringify(segmentPageData, null, 2));
      console.log(
        `  └─ 💾 Data saved for segment ${segmentIndex}: ${path.basename(
          jsonPath
        )}`
      );

      // Add to results array
      pageDataResults.push(segmentPageData);
    }

    await page.close();

    return pageDataResults;
  } catch (error) {
    console.error(`  └─ ❌ Error processing ${url}:`, error);
    throw error;
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

export function getUrlId(url: string): string {
  return url
    .replace(/^https?:\/\//, "")
    .replace(/[^a-zA-Z0-9]/g, "_")
    .substring(0, 50);
}

export async function saveDataToJson(
  data: PageData,
  outputDir: string
): Promise<string> {
  const jsonPath = path.join(outputDir, `${data.id}.json`);

  await fs.writeFile(jsonPath, JSON.stringify(data, null, 2));
  console.log(`  └─ 💾 Data saved: ${path.basename(jsonPath)}`);

  return jsonPath;
}
