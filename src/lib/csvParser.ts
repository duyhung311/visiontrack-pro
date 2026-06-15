import Papa from "papaparse";
import { TrainingLogEntry, PredictionEntry, GroundTruthEntry, ImageResult } from "../types";

export function getTrialBase(filename: string): string {
  if (!filename) return "";
  // Remove suffix extension (e.g. .png, .mat, .jpg, .jpeg)
  let base = filename.replace(/\.[^/.]+$/, "");

  // Sequentially strip out angle-specific suffixes from the very end of the string
  base = base
    .replace(/_images[S|s][0-4]_i[1-2]$/i, "")
    .replace(/_images[S|s][0-4]$/i, "")
    .replace(/images[S|s][0-4]$/i, "")
    .replace(/_S[0-4]$/i, "")
    .replace(/S[0-4]$/i, "")
    .trim()
    .toLowerCase();

  return base;
}

export function extractAngle(filename: string): string {
  if (!filename) return "0";
  // Match patterns like _imagesS0_i1, _imagesS0, S0, s0, _s0, imagesS0, etc.
  const m1 = filename.match(/_images[S|s]([0-4])/i);
  if (m1) return m1[1];
  const m2 = filename.match(/images[S|s]([0-4])/i);
  if (m2) return m2[1];
  const m3 = filename.match(/[S|s]([0-4])/i);
  if (m3) return m3[1];
  return "0";
}

export function parseNumber(val: any, fallback: number | null = null): number | null {
  if (val === undefined || val === null) return fallback;
  const str = String(val).trim();
  if (str === "" || str.toLowerCase() === "none" || str.toLowerCase() === "nan") {
    return fallback;
  }
  const num = Number(str);
  return isNaN(num) ? fallback : num;
}

export function normalizeAngle(angle: any): string {
  if (angle === undefined || angle === null) return "0";
  const str = String(angle).trim().toUpperCase();
  if (str.startsWith("S")) {
    return str.substring(1);
  }
  return str;
}

export function parseTrainingLogs(csvText: string): TrainingLogEntry[] {
  const result = Papa.parse<any>(csvText, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  return result.data
    .map((row) => ({
      epoch: row.epoch ?? row.Epoch ?? 0,
      train_loss: row.train_loss ?? row["train/loss"] ?? row.train_loss ?? 0,
      val_loss: row.val_loss ?? row["val/loss"] ?? row.val_loss ?? 0,
      lr: row.lr ?? row.learning_rate ?? 0,
    }))
    .filter((entry) => entry.epoch > 0 || entry.train_loss > 0 || entry.val_loss > 0);
}

export function parsePredictions(csvText: string): PredictionEntry[] {
  const result = Papa.parse<any>(csvText, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  return result.data
    .map((row) => {
      const cx = parseNumber(row.cx ?? row.CX, 0) ?? 0;
      const cy = parseNumber(row.cy ?? row.CY, 0) ?? 0;
      const cx_coarse = parseNumber(row.cx_coarse ?? row.CX_COARSE, null) ?? undefined;
      const cy_coarse = parseNumber(row.cy_coarse ?? row.CY_COARSE, null) ?? undefined;
      const confidence = parseNumber(row.confidence ?? row.yolo_conf ?? row.CONFIDENCE, null) ?? undefined;
      const azimuth = parseNumber(row.azimuth ?? row.AZIMUTH, null) ?? undefined;
      // In python output, angle is serial number, e.g. 1
      const angle = row.angle !== undefined ? String(row.angle).trim() : undefined;

      return {
        filename: row.filename ?? row.Filename ?? "",
        cx,
        cy,
        cx_coarse,
        cy_coarse,
        confidence,
        azimuth,
        angle,
      };
    })
    .filter((p) => p.filename);
}

export function parseGroundTruth(csvText: string): GroundTruthEntry[] {
  const result = Papa.parse<any>(csvText, {
    header: true,
    dynamicTyping: true,
    skipEmptyLines: true,
  });

  return result.data
    .map((row) => {
      const filename = row.filename ?? row.Filename ?? "";
      const sx = row["comparison image SX"] ?? row["SX"] ?? "";
      const sx_x = parseNumber(row["SX pseudopupil x location"] ?? row["sx_x"], 0) ?? 0;
      const sx_y = parseNumber(row["SX pseudopupil y location"] ?? row["sx_y"], 0) ?? 0;
      
      return {
        filename,
        sx,
        s0_x: parseNumber(row["S0 pseudopupil x location"] ?? row["s0_x"], 0) ?? 0,
        s0_y: parseNumber(row["S0 pseudopupil y location"] ?? row["s0_y"], 0) ?? 0,
        sx_x,
        sx_y,
      };
    })
    .filter((g) => g.filename);
}

export function reconcileDataset(
  predictions: PredictionEntry[],
  groundTruths: GroundTruthEntry[],
  stagedFiles: { name: string; url?: string }[] = []
): ImageResult[] {
  const resultsMap = new Map<string, ImageResult>();

  // Helper helper to register key
  const getOrCreateResult = (trialBase: string, angle: string, originalFilename: string): ImageResult => {
    const key = `${trialBase}::${angle}`;
    if (!resultsMap.has(key)) {
      resultsMap.set(key, {
        key,
        filename: originalFilename,
        displayFilename: originalFilename,
        angle,
      });
    }
    return resultsMap.get(key)!;
  };

  // 1. Process Predictions
  predictions.forEach((pred) => {
    const trialBase = getTrialBase(pred.filename);
    const angle = normalizeAngle(pred.angle ?? extractAngle(pred.filename));
    const res = getOrCreateResult(trialBase, angle, pred.filename);
    res.prediction = pred;
  });

  // 2. Process Ground Truths
  groundTruths.forEach((gt) => {
    const trialBase = getTrialBase(gt.filename);
    const angle = normalizeAngle(gt.sx ?? extractAngle(gt.filename));
    const res = getOrCreateResult(trialBase, angle, gt.filename);
    res.groundTruth = gt;
  });

  // 3. Process staged image/mat files to match them
  stagedFiles.forEach((file) => {
    const trialBase = getTrialBase(file.name);
    const angle = normalizeAngle(extractAngle(file.name));
    const key = `${trialBase}::${angle}`;
    
    // Check if we can map URL to existing result or create new one
    if (resultsMap.has(key)) {
      const res = resultsMap.get(key)!;
      res.imageUrl = file.url;
      res.displayFilename = file.name;
    } else {
      // Ensure we register image assets even if predictions/groundtruth doesn't exist yet for them
      const res = getOrCreateResult(trialBase, angle, file.name);
      res.imageUrl = file.url;
    }
  });

  // Compute errors (Euclidean distance L2) for rows where both exist
  resultsMap.forEach((res) => {
    if (res.prediction && res.groundTruth) {
      const px = res.prediction.cx;
      const py = res.prediction.cy;
      const isS0 = res.angle === "0";
      const tx = isS0 ? res.groundTruth.s0_x : res.groundTruth.sx_x;
      const ty = isS0 ? res.groundTruth.s0_y : res.groundTruth.sx_y;
      
      if (tx > 0 && ty > 0 && px > 0 && py > 0) {
        res.error = Math.sqrt(Math.pow(px - tx, 2) + Math.pow(py - ty, 2));
      }
    }
  });

  return Array.from(resultsMap.values());
}

export function exportPredictionsCSV(results: ImageResult[], overrides: Record<string, { cx: number; cy: number }>): string {
  // azimuth,filename,angle,cx_coarse,cy_coarse,cx,cy,confidence
  const header = "azimuth,filename,angle,cx_coarse,cy_coarse,cx,cy,confidence";
  const rows = results.map((res) => {
    const pred = res.prediction || {
      filename: res.filename,
      cx: 0,
      cy: 0,
      cx_coarse: 0,
      cy_coarse: 0,
      confidence: 1.0,
      azimuth: 0,
      angle: res.angle,
    };

    let cx = pred.cx;
    let cy = pred.cy;
    if (overrides[res.key]) {
      cx = overrides[res.key].cx;
      cy = overrides[res.key].cy;
    }

    return [
      pred.azimuth ?? 0,
      pred.filename ?? res.filename,
      pred.angle ?? `S${res.angle}`,
      pred.cx_coarse ?? cx,
      pred.cy_coarse ?? cy,
      cx,
      cy,
      pred.confidence ?? 1.0
    ].join(",");
  });

  return [header, ...rows].join("\n");
}

export function exportGroundTruthCSV(results: ImageResult[], overrides: Record<string, { cx: number; cy: number }>): string {
  // filename,comparison image SX,SX pseudopupil x location,SX pseudopupil y location
  const header = "filename,comparison image SX,SX pseudopupil x location,SX pseudopupil y location";
  const rows = results.map((res) => {
    const gt = res.groundTruth || {
      filename: res.filename,
      sx: `S${res.angle}`,
      s0_x: 0,
      s0_y: 0,
      sx_x: 0,
      sx_y: 0,
    };

    let sx_x = gt.sx_x;
    let sx_y = gt.sx_y;
    // Overriding is for predictions normally, but we can also use custom overrides to update sx locations if needed
    if (overrides[res.key] && res.angle !== "0") {
      sx_x = overrides[res.key].cx;
      sx_y = overrides[res.key].cy;
    }

    return [
      gt.filename ?? res.filename,
      gt.sx || `S${res.angle}`,
      sx_x,
      sx_y
    ].join(",");
  });

  return [header, ...rows].join("\n");
}
