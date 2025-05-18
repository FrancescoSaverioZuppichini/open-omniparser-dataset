// Define export interfaces for our data structures
export interface ElementBoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface InteractiveElement {
  type: string; // HTML tag name (button, a, select, etc.)
  text: string; // Text content of the element
  boundingBox: ElementBoundingBox; // Position and dimensions
  attributes: Record<string, string>; // Additional attributes like id, class, etc.
  isVisible: boolean; // Whether the element is visible
  xpath: string; // XPath for the element
}

export interface PageData {
  url: string;
  id: string;
  viewport: {
    width: number;
    height: number;
  };
  pageHeight: number;
  pageWidth: number;
  screenshotPath: string;
  elements: InteractiveElement[];
}

export interface Config {
  selectors: string[];
}

export type WebsitesConfig = {
  name: string;
  pages: [
    {
      domain: string;
      urls: string[];
      preprocessing?: {
        deleteQueries?: string[];
        cookieQuery?: string;
      };
    }
  ];
}[];
