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
  page: { height: number; width: number };
  screenshotPath: string;
  elements: InteractiveElement[];
}

export interface Config {
  viewport: { height: number; width: number };
  screenshot: { quality: number };
}

export type WebsitesConfig = {
  name: string;
  pages: {
    domain: string;
    urls: string[];
    preprocessing?: {
      deleteQueries?: string[];
      cookieQuery?: string;
    };
  }[];
}[];

export interface CocoDataset {
  images: CocoImage[];
  annotations: CocoAnnotation[];
  categories: CocoCategory[];
}

export interface CocoImage {
  id: number;
  width: number;
  height: number;
  file_name: string;
}

export interface CocoAnnotation {
  id: number;
  image_id: number;
  category_id: number;
  bbox: [number, number, number, number]; // [x, y, width, height]
  area?: number; // Can be computed from bbox if not provided
  iscrowd?: 0 | 1; // Optional flag (0 for individual objects, 1 for groups)
}

export interface CocoCategory {
  id: number;
  name: string;
  supercategory?: string;
}
