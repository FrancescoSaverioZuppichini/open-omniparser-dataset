import type { Config } from "./types";

const config: Config = {
  selectors: [
    "button",
    "a",
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
  ],
};

export default config;
