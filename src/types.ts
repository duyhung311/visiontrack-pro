export interface FileInfo {
  name: string;
  size: number;
  mtime: string;
  type: "mat" | "json" | "image" | "other";
}

export interface TrainingStatus {
  percentage: number;
  currentEpoch: number;
  totalEpochs: number;
  metrics: string;
  status: "idle" | "running" | "completed" | "aborted" | "failed";
  config: {
    version: string;
    size: string;
    epochs: number;
    batch: number;
    device: string;
  } | null;
}

export interface InferenceStatus {
  percentage: number;
  currentFile: number;
  totalFiles: number;
  currentFileName: string;
  status: "idle" | "running" | "completed" | "aborted" | "failed";
  config: {
    version: string;
    size: string;
    device: string;
    outputFilename: string;
  } | null;
}

export interface RunHistoryItem {
  id: string;
  modelConfig: string; // e.g., "yolo8m"
  version: string;
  size: string;
  timestamp: string;
  status: string;
  weightsFile: string | null;
  associatedCSV: string | null;
}

export interface CSVOutputItem {
  filename: string;
  size: number;
  timestamp: string;
}

export interface TrainingLogEntry {
  epoch: number;
  train_loss: number;
  val_loss: number;
  lr: number;
}

export interface PredictionEntry {
  filename: string;
  cx: number;
  cy: number;
  cx_coarse?: number;
  cy_coarse?: number;
  confidence?: number;
  azimuth?: number;
  angle?: string | number; // Corresponds to node codes, e.g., 0, 1, 2, 3, 4
}

export interface GroundTruthEntry {
  filename: string;
  sx: string;    // Active eye nodes, e.g. "S1", "S2"
  s0_x: number;
  s0_y: number;
  sx_x: number;
  sx_y: number;
}

export interface ImageResult {
  key: string;              // Compound internal key: `${trialBase}::${normalizedAngle}`
  filename: string;         // Underlying raw reference filename
  displayFilename: string;  // Formatted filename, e.g. `20260127_..._site_1_S3.mat`
  prediction?: PredictionEntry;
  groundTruth?: GroundTruthEntry;
  error?: number;           // L2 Euclidean distance (px) between model & reference point
  isCustom?: boolean;       // Set to true if has active user overrides
  imageUrl?: string;        // Local blob URL matching the active image format
  angle: string;            // Active node suffix (e.g. '0', '1', '2', '3', '4')
}
