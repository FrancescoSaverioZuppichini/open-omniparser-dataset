import { Browser, Page, ElementHandle } from "puppeteer";
import puppeteer from "puppeteer-extra";
import fs from "fs/promises";
import path from "path";
import type { InteractiveElement, PageData } from "./types";
import StealthPlugin from "puppeteer-extra-plugin-stealth";
import AdblockerPlugin from "puppeteer-extra-plugin-adblocker";

const OMNI_DATASET_DIR = process.env.OMNI_DATASET_DIR ?? "../dataset";

async function acceptCookieConsent(
  page: Page,
  timeout: number = 1000
): Promise<boolean> {
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
  ];

  try {
    // Try each selector until one works
    for (const selector of selectors) {
      try {
        // Wait for selector with a short timeout
        const button = await page.waitForSelector(selector, { timeout });

        if (button) {
          // Click the button and wait a moment
          await button.click();
          // Use this instead of waitForTimeout
          await page.evaluate(
            () => new Promise((resolve) => setTimeout(resolve, 500))
          );
          console.log(
            `Successfully clicked cookie consent button: ${selector}`
          );
          return true;
        }
      } catch (e) {
        // This selector wasn't found - continue to next one
      }
    }

    // If execution reaches here, no selectors were found
    console.log("No cookie consent buttons found");
    return false;
  } catch (error) {
    console.error("Error while trying to accept cookies:", error);
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

async function capturePageData({
  url,
  deleteQuery,
  outputDir,
  cookie,
}: {
  url: string;
  deleteQuery?: string;
  cookie?: string;
  outputDir: string;
}): Promise<PageData> {
  let browser: Browser | null = null;

  try {
    // Create output directory if it doesn't exist
    await fs.mkdir(outputDir, { recursive: true });

    puppeteer.use(StealthPlugin());
    puppeteer.use(AdblockerPlugin({ blockTrackers: true }));

    // Launch the browser
    browser = await puppeteer.launch({
      headless: true,
    });

    const page: Page = await browser.newPage();

    // Set viewport
    const viewport = {
      width: 1280,
      height: 800,
    };
    await page.setViewport(viewport);

    console.log(`Navigating to ${url}...`);
    await page.goto(url, {
      waitUntil: "networkidle2",
      timeout: 60000,
    });
    await acceptCookieConsent(page);

    // Remove elements matching deleteQuery if provided
    if (deleteQuery) {
      console.log(`Removing elements matching selector: ${deleteQuery}`);
      await page.evaluate((selector) => {
        const elements = document.querySelectorAll(selector);
        elements.forEach((element) => {
          element.remove();
        });
      }, deleteQuery);
    }

    if (cookie) {
      console.log(`Clicking on cookie using selector: ${cookie}`);
      try {
        await page.click(cookie);
      } catch (error) {
        console.error(
          `Failed to click on element with selector "${cookie}":`,
          error
        );
      }
    }

    // Get page dimensions
    const dimensions = await page.evaluate(() => {
      return {
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
      };
    });

    // Take screenshot
    const urlSlug = url
      .replace(/^https?:\/\//, "")
      .replace(/[^a-zA-Z0-9]/g, "_")
      .substring(0, 50);
    // const timestamp = new Date()
    //   .toISOString()
    //   .replace(/:/g, "-")
    //   .replace(/\..+/, "");
    const screenshotPath = path.join(outputDir, `${urlSlug}.png`);

    console.log(`Taking screenshot and saving to ${screenshotPath}...`);
    await page.screenshot({
      path: screenshotPath,
      fullPage: true,
    });

    // Capture interactive elements
    console.log("Capturing interactive elements...");
    const elements = await captureInteractiveElements(page);

    return {
      url,
      viewport,
      pageHeight: dimensions.pageHeight,
      pageWidth: dimensions.pageWidth,
      screenshotPath,
      elements,
    };
  } catch (error) {
    console.error("Error during page data capture:", error);
    throw error;
  } finally {
    if (browser) {
      await browser.close();
      console.log("Browser closed.");
    }
  }
}

async function saveDataToJson(
  data: PageData,
  outputDir: string
): Promise<string> {
  const urlSlug = data.url
    .replace(/^https?:\/\//, "")
    .replace(/[^a-zA-Z0-9]/g, "_")
    .substring(0, 50);
  const jsonPath = path.join(outputDir, `${urlSlug}.json`);

  await fs.writeFile(jsonPath, JSON.stringify(data, null, 2));
  console.log(`Data saved to ${jsonPath}`);

  return jsonPath;
}

// Run with Bun
(async () => {
  const start = performance.now();
  const outputDir = OMNI_DATASET_DIR;
  // const url = "https://commoncrawl.org/get-started";
  // const url = "https://github.com/KwaiVGI/LivePortrait";
  // const url =
  //   "https://github.com/KwaiVGI/LivePortrait/commit/6c4a883a9e67330fdecb0982b0c0611d425c8681";
  const url = "https://www.facebook.com/marketplace";
  const deleteQuery = ".__fb-light-mode.x1n2onr6.xzkaem6";
  const cookie = '[aria-label="Allow all cookies"]';

  try {
    console.log(`Starting data collection for ${url}`);
    const pageData = await capturePageData({
      url,
      deleteQuery,
      cookie,
      outputDir,
    });

    console.log(`Found ${pageData.elements.length} interactive elements`);
    await saveDataToJson(pageData, outputDir);

    console.log("Data collection completed successfully!");
  } catch (error) {
    console.error("Data collection process failed:", error);
    process.exit(1);
  }

  console.log(`${(performance.now() - start) / 1000}s`);
})();
