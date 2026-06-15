import React, { useState, useEffect, useRef, useMemo } from "react";
import { 
  Eye, 
  Settings, 
  Terminal, 
  Upload, 
  Save, 
  HelpCircle, 
  ArrowLeft, 
  ArrowRight, 
  Table, 
  Download, 
  Undo, 
  Info, 
  Sliders, 
  ChevronLeft, 
  ChevronRight, 
  Check, 
  Search, 
  TrendingUp, 
  AlertTriangle, 
  Layers, 
  FileSpreadsheet,
  Brain,
  Copy
} from "lucide-react";
import { 
  ResponsiveContainer, 
  LineChart, 
  CartesianGrid, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Legend, 
  Line 
} from "recharts";

import { 
  PredictionEntry, 
  GroundTruthEntry, 
  TrainingLogEntry, 
  ImageResult, 
  CSVOutputItem,
  FileInfo
} from "../types";

import { 
  parsePredictions, 
  parseGroundTruth, 
  parseTrainingLogs,
  exportPredictionsCSV, 
  exportGroundTruthCSV, 
  getTrialBase,
  extractAngle,
  normalizeAngle
} from "../lib/csvParser";

interface VisionTrackInspectorPanelProps {
  activeTab: string;
  setActiveTab: (tab: "dashboard" | "inspector" | "logs") => void;

  predictionEntries: PredictionEntry[];
  setPredictionEntries: (entries: PredictionEntry[]) => void;
  predictionsFilename?: string;
  setPredictionsFilename?: (name: string) => void;
  groundTruthEntries: GroundTruthEntry[];
  setGroundTruthEntries: (entries: GroundTruthEntry[]) => void;
  trainingLogEntries: TrainingLogEntry[];
  setTrainingLogEntries: (entries: TrainingLogEntry[]) => void;

  reconciledResults: ImageResult[];
  selectedResultKey: string | null;
  setSelectedResultKey: (key: string | null) => void;
  selectedResult: ImageResult | null;

  modifiedPredictions: Record<string, { cx: number; cy: number }>;
  setModifiedPredictions: React.Dispatch<React.SetStateAction<Record<string, { cx: number; cy: number }>>>;
  modifiedCoarse: Record<string, { cx: number; cy: number }>;
  setModifiedCoarse: React.Dispatch<React.SetStateAction<Record<string, { cx: number; cy: number }>>>;

  invertY: boolean;
  setInvertY: (invert: boolean) => void;
  yOffset: number;
  setYOffset: (offset: number) => void;
  showCoarse: boolean;
  setShowCoarse: (show: boolean) => void;
  showGroundTruth: boolean;
  setShowGroundTruth: (show: boolean) => void;
  showPredictions: boolean;
  setShowPredictions: (show: boolean) => void;
  showGrid: boolean;
  setShowGrid: (show: boolean) => void;
  csvOutputs: CSVOutputItem[];
  files: FileInfo[];
  refreshFiles: () => Promise<void>;
  localImageFiles?: { name: string; size: number; url: string }[];
  setLocalImageFiles?: React.Dispatch<React.SetStateAction<{ name: string; size: number; url: string }[]>>;
  availableDirs?: string[];
  selectedDir?: string;
  setSelectedDir?: (dir: string) => void;
}

export default function VisionTrackInspectorPanel({
  activeTab,
  setActiveTab,

  predictionEntries,
  setPredictionEntries,
  predictionsFilename = "",
  setPredictionsFilename,
  groundTruthEntries,
  setGroundTruthEntries,
  trainingLogEntries,
  setTrainingLogEntries,

  reconciledResults,
  selectedResultKey,
  setSelectedResultKey,
  selectedResult,

  modifiedPredictions,
  setModifiedPredictions,
  modifiedCoarse,
  setModifiedCoarse,

  invertY,
  setInvertY,
  yOffset,
  setYOffset,
  showCoarse,
  setShowCoarse,
  showGroundTruth,
  setShowGroundTruth,
  showPredictions,
  setShowPredictions,
  showGrid,
  setShowGrid,
  csvOutputs,
  files,
  refreshFiles,
  localImageFiles,
  setLocalImageFiles,
  availableDirs = [],
  selectedDir = "Default Workspace",
  setSelectedDir
}: VisionTrackInspectorPanelProps) {
  
  // Custom Filters & Search Local States
  const [fileFilter, setFileFilter] = useState("");
  const [lowAccuracyFilter, setLowAccuracyFilter] = useState(false);
  const [lowAccuracyThreshold, setLowAccuracyThreshold] = useState(5.0);

  // States for drop box loaders / servers
  const [selectedServerCsv, setSelectedServerCsv] = useState("");
  const [panelMessage, setPanelMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Drag State Tracker
  const [dragPoint, setDragPoint] = useState<"fine" | null>(null);
  const containerRef = useRef<SVGSVGElement | null>(null);

  // Image upload and drag-and-drop state/handlers
  const [imageDragActive, setImageDragActive] = useState(false);
  const imageInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setImageDragActive(true);
    } else if (e.type === "dragleave") {
      setImageDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setImageDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const filesArr = Array.from(e.dataTransfer.files) as File[];
      const isImg = (file: File) => 
        file.type.startsWith("image/") || 
        file.name.toLowerCase().endsWith(".png") || 
        file.name.toLowerCase().endsWith(".jpg") || 
        file.name.toLowerCase().endsWith(".jpeg");
      const imageFilesArr = filesArr.filter(isImg);

      if (imageFilesArr.length === 0) {
        setPanelMessage({ type: "error", text: "Please supply valid image files (PNG, JPG, JPEG) to upload." });
        return;
      }
      uploadImageFiles(imageFilesArr);
    }
  };

  const handleImageFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const filesArr = Array.from(e.target.files) as File[];
      uploadImageFiles(filesArr);
    }
  };

  const uploadImageFiles = async (filesToUpload: File[]) => {
    setPanelMessage({ type: "success", text: `Uploading and caching ${filesToUpload.length} image file(s)...` });
    
    // Instantly register on-the-fly client-side object URLs for preview/inspecting
    if (setLocalImageFiles) {
      setLocalImageFiles(prev => {
        const next = [...prev];
        filesToUpload.forEach(file => {
          if (!next.some(x => x.name === file.name)) {
            next.push({
              name: file.name,
              size: file.size,
              url: URL.createObjectURL(file)
            });
          }
        });
        return next;
      });
    }

    try {
      let succeededCount = 0;
      let failedCount = 0;

      const uploadPromises = filesToUpload.map(async (file) => {
        const singleFormData = new FormData();
        singleFormData.append("files", file);
        try {
          const resp = await fetch("/api/upload", {
            method: "POST",
            body: singleFormData,
          });
          if (resp.ok) {
            succeededCount++;
          } else {
            failedCount++;
          }
        } catch (err) {
          failedCount++;
        }
      });

      await Promise.all(uploadPromises);

      if (failedCount === 0) {
        setPanelMessage({ type: "success", text: `Successfully matched and uploaded all ${succeededCount} image file(s)!` });
      } else {
        setPanelMessage({ 
          type: "warning", 
          text: `Matched ${succeededCount} image(s) locally. ${failedCount} server upload(s) failed.` 
        });
      }
      await refreshFiles();
    } catch (err: any) {
      setPanelMessage({ type: "error", text: `Upload error: ${err.message}` });
    }
  };

  // Auto Dismiss messages
  useEffect(() => {
    if (panelMessage) {
      const timer = setTimeout(() => setPanelMessage(null), 6000);
      return () => clearTimeout(timer);
    }
  }, [panelMessage]);

  // Handle Predictions Upload from sidebar
  const handlePredictionsUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      try {
        const parsed = parsePredictions(text);
        if (parsed.length === 0) {
          throw new Error("No coordinate prediction entry records mapped from this file");
        }
        setPredictionEntries(parsed);
        if (setPredictionsFilename) {
          setPredictionsFilename(file.name);
        }
        setPanelMessage({ type: "success", text: `Imported ${parsed.length} prediction node points successfully!` });
        
        // Auto-select first loaded item
        const base = getTrialBase(parsed[0].filename);
        const angle = normalizeAngle(parsed[0].angle ?? extractAngle(parsed[0].filename));
        setSelectedResultKey(`${base}::${angle}`);
      } catch (err: any) {
        setPanelMessage({ type: "error", text: `Predictions parse failed: ${err.message}` });
      }
    };
    reader.readAsText(file);
  };

  // Handle Ground Truth benchmark upload
  const handleGroundTruthUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      try {
        const parsed = parseGroundTruth(text);
        if (parsed.length === 0) {
          throw new Error("Empty dataset ground truth records.");
        }
        setGroundTruthEntries(parsed);
        setPanelMessage({ type: "success", text: `Loaded ${parsed.length} truth benchmark items successfully!` });
      } catch (err: any) {
        setPanelMessage({ type: "error", text: `Ground Truth parse failed: ${err.message}` });
      }
    };
    reader.readAsText(file);
  };

  // Handle Training Loss Log upload
  const handleTrainingLogUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      try {
        const parsed = parseTrainingLogs(text);
        setTrainingLogEntries(parsed);
        setPanelMessage({ type: "success", text: `Successfully loaded metrics for ${parsed.length} epochs!` });
      } catch (err: any) {
        setPanelMessage({ type: "error", text: `Training Log parse failed: ${err.message}` });
      }
    };
    reader.readAsText(file);
  };

  // Load select local storage server outputs
  const handleLoadSelectedServerCsv = async () => {
    if (!selectedServerCsv) return;
    try {
      const resp = await fetch(`/api/csv-output/${encodeURIComponent(selectedServerCsv)}`);
      if (!resp.ok) {
        throw new Error(`Server returned status code: ${resp.status}`);
      }
      const text = await resp.text();
      const parsed = parsePredictions(text);
      if (parsed.length === 0) {
        throw new Error("Resource is empty or lacks correctly formatted coordinates.");
      }
      setPredictionEntries(parsed);
      if (setPredictionsFilename) {
        setPredictionsFilename(selectedServerCsv);
      }
      setPanelMessage({ type: "success", text: `Successfully restored ${parsed.length} items from server-side local storage!` });

      // Auto-switch to matching dataset directory if selectedServerCsv lies in subdirectory
      if (selectedServerCsv.includes("/") && setSelectedDir) {
        const folder = selectedServerCsv.split("/")[0];
        if (availableDirs.includes(folder)) {
          setSelectedDir(folder);
        }
      }
      
      const base = getTrialBase(parsed[0].filename);
      const angle = normalizeAngle(parsed[0].angle ?? extractAngle(parsed[0].filename));
      setSelectedResultKey(`${base}::${angle}`);
    } catch (err: any) {
      setPanelMessage({ type: "error", text: `Failed to fetch server output: ${err.message}` });
    }
  };

  // Automatic Natural Size detector
  const [naturalSize, setNaturalSize] = useState({ width: 2048, height: 1536 });
  const [plotDotRadius, setPlotDotRadius] = useState(1.0);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [offsetStart, setOffsetStart] = useState({ x: 0, y: 0 });
  const [hasMoved, setHasMoved] = useState(false);

  useEffect(() => {
    if (selectedResult?.imageUrl) {
      const img = new Image();
      img.src = selectedResult.imageUrl;
      img.onload = () => {
        if (img.naturalWidth && img.naturalHeight) {
          setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
        }
      };
    } else {
      // Default fallback dimensions for calculations
      setNaturalSize({ width: 2048, height: 1536 });
    }
  }, [selectedResult?.imageUrl]);

  // Transform coordinates factoring in Y inversion, Y offset, and resolution downscaling
  const getRenderCoords = (cx: number, cy: number) => {
    const nWidth = naturalSize.width || 640;
    const nHeight = naturalSize.height || 480;
    
    // Scale from natural size to 640x480 SVG display box
    const sx = cx * (640 / nWidth);
    const sy = cy * (480 / nHeight);
    
    const renderedY = invertY ? 480 - sy + yOffset : sy + yOffset;
    return { x: sx, y: renderedY };
  };

  const getSavedFromRender = (rx: number, ry: number) => {
    const nWidth = naturalSize.width || 640;
    const nHeight = naturalSize.height || 480;
    
    // Adjust y using inverted state
    const sy = invertY ? 480 - ry + yOffset : ry - yOffset;
    
    // Scale back up from 640x480 box to original natural image coordinates
    const cx = rx * (nWidth / 640);
    const cy = sy * (nHeight / 480);
    return { cx, cy };
  };

  const getCanvasCoords = (clientX: number, clientY: number) => {
    if (!containerRef.current) return { rx: 0, ry: 0 };
    const rect = containerRef.current.getBoundingClientRect();
    
    // Convert click position to the standard 640x480 canvas box
    const baseSvgX = ((clientX - rect.left) / rect.width) * 640;
    const baseSvgY = ((clientY - rect.top) / rect.height) * 480;

    // Convert back from current zoom and pan translation coordinates to standard 640x480
    const rx = (baseSvgX - offset.x) / zoom;
    const ry = (baseSvgY - offset.y) / zoom;

    return { rx, ry };
  };

  // Selected result coordinate points
  const fineX = selectedResult
    ? (modifiedPredictions[selectedResult.key]?.cx ?? (selectedResult.prediction?.cx !== undefined && selectedResult.prediction?.cx !== null && selectedResult.prediction.cx > 0 ? selectedResult.prediction.cx : null))
    : null;
  const fineY = selectedResult
    ? (modifiedPredictions[selectedResult.key]?.cy ?? (selectedResult.prediction?.cy !== undefined && selectedResult.prediction?.cy !== null && selectedResult.prediction.cy > 0 ? selectedResult.prediction.cy : null))
    : null;

  const isS0 = selectedResult?.angle === "0";
  const rawGtX = selectedResult?.groundTruth
    ? (isS0 ? selectedResult.groundTruth.s0_x : selectedResult.groundTruth.sx_x)
    : null;
  const rawGtY = selectedResult?.groundTruth
    ? (isS0 ? selectedResult.groundTruth.s0_y : selectedResult.groundTruth.sx_y)
    : null;
  const gtX = (rawGtX !== null && rawGtX > 0) ? rawGtX : null;
  const gtY = (rawGtY !== null && rawGtY > 0) ? rawGtY : null;

  // L2 Euclidean absolute displacement error
  const displacementError = useMemo(() => {
    if (fineX !== null && fineY !== null && gtX !== null && gtY !== null) {
      return Math.sqrt(Math.pow(fineX - gtX, 2) + Math.pow(fineY - gtY, 2));
    }
    return null;
  }, [fineX, fineY, gtX, gtY]);

  // Stats computed dynamically
  const stats = useMemo(() => {
    const overridesCount = Object.keys(modifiedPredictions).length;
    const reconciledCount = reconciledResults.length;
    const matchesCount = reconciledResults.filter((r) => r.prediction && r.groundTruth).length;
    
    let sumErr = 0;
    let withErrCount = 0;
    let maxErrorValue = 0;

    reconciledResults.forEach((r) => {
      const px = modifiedPredictions[r.key]?.cx ?? r.prediction?.cx ?? null;
      const py = modifiedPredictions[r.key]?.cy ?? r.prediction?.cy ?? null;
      const isS0_r = r.angle === "0";
      const gx = r.groundTruth
        ? (isS0_r ? r.groundTruth.s0_x : r.groundTruth.sx_x)
        : null;
      const gy = r.groundTruth
        ? (isS0_r ? r.groundTruth.s0_y : r.groundTruth.sx_y)
        : null;

      if (px !== null && py !== null && gx !== null && gy !== null) {
        const err = Math.sqrt(Math.pow(px - gx, 2) + Math.pow(py - gy, 2));
        sumErr += err;
        withErrCount++;
        if (err > maxErrorValue) {
          maxErrorValue = err;
        }
      }
    });

    return {
      overridesCount,
      reconciledCount,
      matchesCount,
      avgErr: withErrCount > 0 ? sumErr / withErrCount : null,
      maxErr: maxErrorValue
    };
  }, [reconciledResults, modifiedPredictions]);

  // Filter dataset results inside list
  const filteredReconciledResults = useMemo(() => {
    return reconciledResults.filter((r) => {
      if (fileFilter) {
        const query = fileFilter.toLowerCase();
        const base = getTrialBase(r.filename).toLowerCase();
        const full = r.filename.toLowerCase();
        if (!base.includes(query) && !full.includes(query)) {
          return false;
        }
      }

      if (lowAccuracyFilter) {
        const px = modifiedPredictions[r.key]?.cx ?? r.prediction?.cx ?? null;
        const py = modifiedPredictions[r.key]?.cy ?? r.prediction?.cy ?? null;
        const isS0_r = r.angle === "0";
        const gx = r.groundTruth
          ? (isS0_r ? r.groundTruth.s0_x : r.groundTruth.sx_x)
          : null;
        const gy = r.groundTruth
          ? (isS0_r ? r.groundTruth.s0_y : r.groundTruth.sx_y)
          : null;

        if (px !== null && py !== null && gx !== null && gy !== null) {
          const err = Math.sqrt(Math.pow(px - gx, 2) + Math.pow(py - gy, 2));
          if (err <= lowAccuracyThreshold) {
            return false;
          }
        } else {
          return false; // exclude if no ground truth comparison
        }
      }

      return true;
    });
  }, [reconciledResults, fileFilter, lowAccuracyFilter, lowAccuracyThreshold, modifiedPredictions]);

  // Click & Drag Canvas Interactions
  const handlePointerDown = (e: React.PointerEvent<SVGElement>, pointType: "fine") => {
    e.stopPropagation();
    e.preventDefault();
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
    setDragPoint(pointType);
  };

  const handleSvgPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (dragPoint) return;
    e.preventDefault();
    setIsPanning(true);
    setDragStart({ x: e.clientX, y: e.clientY });
    setOffsetStart({ x: offset.x, y: offset.y });
    setHasMoved(false);
  };

  const handlePointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (dragPoint && selectedResult) {
      // Scale client coordinate offsets back to standard coordinates
      const { rx, ry } = getCanvasCoords(e.clientX, e.clientY);
      
      // Enforce layout limits
      const boundedX = Math.max(0, Math.min(640, rx));
      const boundedY = Math.max(0, Math.min(480, ry));

      const saved = getSavedFromRender(boundedX, boundedY);

      setModifiedPredictions((prev) => ({
        ...prev,
        [selectedResult.key]: { cx: saved.cx, cy: saved.cy }
      }));
    } else if (isPanning) {
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      
      // Translate delta in pixel coordinates to SVG canvas viewBox scale
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        const svgDx = (dx / rect.width) * 640;
        const svgDy = (dy / rect.height) * 480;
        
        setOffset({
          x: offsetStart.x + svgDx,
          y: offsetStart.y + svgDy
        });
        
        if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
          setHasMoved(true);
        }
      }
    }
  };

  const handlePointerUp = (e: React.PointerEvent<SVGSVGElement>) => {
    if (dragPoint) {
      (e.target as Element).releasePointerCapture(e.pointerId);
      setDragPoint(null);
    }
    
    if (isPanning) {
      setIsPanning(false);
      
      // If the cursor was clicked but not moved, treat as Single Click to Point!
      if (!hasMoved && selectedResult) {
        const { rx, ry } = getCanvasCoords(e.clientX, e.clientY);
        
        const boundedX = Math.max(0, Math.min(640, rx));
        const boundedY = Math.max(0, Math.min(480, ry));

        const saved = getSavedFromRender(boundedX, boundedY);

        setModifiedPredictions((prev) => ({
          ...prev,
          [selectedResult.key]: { cx: saved.cx, cy: saved.cy }
        }));
        
        setPanelMessage({ 
          type: "success", 
          text: `Prediction centroid pointed/centered at (${saved.cx.toFixed(1)}, ${saved.cy.toFixed(1)})` 
        });
      }
    }
  };

  const handleWheel = (e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault();
    const zoomFactor = 1.15;
    let nextZoom = zoom;
    if (e.deltaY < 0) {
      nextZoom = Math.min(15, zoom * zoomFactor);
    } else {
      nextZoom = Math.max(0.5, zoom / zoomFactor);
    }

    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const svgX = (mouseX / rect.width) * 640;
      const svgY = (mouseY / rect.height) * 480;

      setOffset((prev) => {
        const dx = svgX - prev.x;
        const dy = svgY - prev.y;
        return {
          x: svgX - dx * (nextZoom / zoom),
          y: svgY - dy * (nextZoom / zoom),
        };
      });
    }
    setZoom(nextZoom);
  };

  const handleDoubleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    e.preventDefault();
    setZoom(1);
    setOffset({ x: 0, y: 0 });
    setPanelMessage({ type: "success", text: "Visual inspector view reset to default (1.0x)." });
  };

  const handleResetActiveOverride = () => {
    if (!selectedResult) return;
    setModifiedPredictions((prev) => {
      const copy = { ...prev };
      delete copy[selectedResult.key];
      return copy;
    });
    setPanelMessage({ type: "success", text: "Draggable predictions coordinate override cleared." });
  };

  const handleTriggerExportPredictions = () => {
    try {
      const csvContent = exportPredictionsCSV(reconciledResults, modifiedPredictions);
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `predictions_export_${Date.now()}.csv`);
      link.style.display = "none";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setPanelMessage({ type: "success", text: "Staged coordinate exports processed and downloaded as CSV file successfully!" });
    } catch (err: any) {
      setPanelMessage({ type: "error", text: `Export failed: ${err.message}` });
    }
  };

  const handleTriggerExportGroundTruth = () => {
    try {
      const csvContent = exportGroundTruthCSV(reconciledResults, modifiedPredictions);
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `ground_truth_export_${Date.now()}.csv`);
      link.style.display = "none";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setPanelMessage({ type: "success", text: "Benchmark dataset values exported and downloaded successfully." });
    } catch (err: any) {
      setPanelMessage({ type: "error", text: `Ground truth export error: ${err.message}` });
    }
  };

  // Header quick jump triggers
  const handleSelectedResultPrev = () => {
    if (filteredReconciledResults.length === 0) return;
    const currentIndex = filteredReconciledResults.findIndex((r) => r.key === selectedResultKey);
    if (currentIndex > 0) {
      setSelectedResultKey(filteredReconciledResults[currentIndex - 1].key);
    } else {
      setSelectedResultKey(filteredReconciledResults[filteredReconciledResults.length - 1].key);
    }
  };

  const handleSelectedResultNext = () => {
    if (filteredReconciledResults.length === 0) return;
    const currentIndex = filteredReconciledResults.findIndex((r) => r.key === selectedResultKey);
    if (currentIndex >= 0 && currentIndex < filteredReconciledResults.length - 1) {
      setSelectedResultKey(filteredReconciledResults[currentIndex + 1].key);
    } else {
      setSelectedResultKey(filteredReconciledResults[0].key);
    }
  };

  const handleOverrideAndSave = () => {
    if (!selectedResult) return;
    setPanelMessage({
      type: "success",
      text: `Successfully staged centroid override coordinates for trial '${getTrialBase(selectedResult.filename)}'. Click 'Export Predictions' to compile changes.`
    });
  };

  const getBadgeTag = () => {
    if (!selectedResult) return null;
    if (selectedResult.prediction && selectedResult.groundTruth) {
      return (
        <span className="bg-emerald-100 text-emerald-800 text-[10px] font-bold px-2 py-0.5 rounded font-mono uppercase">
          Matched Comparison
        </span>
      );
    }
    if (selectedResult.prediction) {
      return (
        <span className="bg-amber-100 text-amber-700 text-[10px] font-bold px-2 py-0.5 rounded font-mono uppercase">
          Prediction Only
        </span>
      );
    }
    if (selectedResult.groundTruth) {
      return (
        <span className="bg-blue-100 text-blue-700 text-[10px] font-bold px-2 py-0.5 rounded font-mono uppercase">
          Ground Truth Only
        </span>
      );
    }
    return (
      <span className="bg-slate-100 text-slate-500 text-[10px] font-medium px-2 py-0.5 rounded font-mono">
        Staged Asset
      </span>
    );
  };

  return (
    <div className="flex h-screen max-h-screen overflow-hidden text-slate-800 bg-[#f8fafc]">
      
      {/* 1. PERSISTENT LEFT COLUMN: NAVIGATION & DATA INTAKE FEED */}
      <div className="w-[310px] border-r border-slate-200 bg-white h-full flex flex-col flex-shrink-0 select-none">
        
        {/* Title / Brand Section */}
        <div className="p-5 border-b border-slate-100">
          <div className="text-[10px] font-mono font-black tracking-wider text-indigo-600 uppercase">VISIONTRACK PRO</div>
          <div className="text-base font-black tracking-tight text-slate-900 mt-1 flex items-center gap-1.5 leading-none">
            <Brain className="w-5 h-5 text-indigo-600" />
            Mission Control
          </div>
        </div>

        {/* Navigation Sidebar */}
        <div className="p-3 border-b border-slate-100 flex flex-col gap-1">
          <button
            onClick={() => setActiveTab("dashboard")}
            className={`w-full flex items-center gap-3 px-3 py-2 text-xs font-semibold rounded-lg transition-all text-left ${
              activeTab === "dashboard"
                ? "bg-slate-100 text-indigo-700 shadow-3xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
            }`}
          >
            <Settings className="w-4 h-4 text-slate-400" />
            DASHBOARD STATION
          </button>
          
          <button
            onClick={() => setActiveTab("inspector")}
            className={`w-full flex items-center gap-3 px-3 py-2 text-xs font-semibold rounded-lg transition-all text-left ${
              activeTab === "inspector"
                ? "bg-indigo-50/70 text-indigo-700 font-extrabold border-l-2 border-indigo-600 pl-2.5"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
            }`}
          >
            <Eye className="w-4 h-4 text-indigo-600" />
            INSPECTOR PANEL
          </button>

          <button
            onClick={() => setActiveTab("logs")}
            className={`w-full flex items-center gap-3 px-3 py-2 text-xs font-semibold rounded-lg transition-all text-left ${
              activeTab === "logs"
                ? "bg-slate-100 text-indigo-700 shadow-3xs"
                : "text-slate-600 hover:text-slate-900 hover:bg-slate-50"
            }`}
          >
            <Terminal className="w-4 h-4 text-slate-400" />
            LOGS FEED
          </button>
        </div>

        {/* Scrollable control settings panel */}
        <div className="flex-1 overflow-y-auto p-4 space-y-5 custom-scrollbar">
          
          {/* LOADED FILES box */}
          <div className="bg-slate-50 rounded-xl p-3 border border-slate-200">
            <h4 className="text-[10px] font-mono font-black text-slate-400 uppercase tracking-wider mb-2.5">LOADED FILES</h4>
            <div className="space-y-1.5 text-[11px] font-mono">
              <div className="flex items-center justify-between">
                <span className="text-slate-500 font-medium">Predictions:</span>
                <span className={`font-bold truncate max-w-[140px] text-right ${predictionEntries.length > 0 ? "text-emerald-600" : "text-slate-400"}`}>
                  {predictionEntries.length > 0 ? `Loaded (${predictionEntries.length} pts)` : "Missing"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500 font-medium">Ground Truth:</span>
                <span className={`font-bold text-right ${groundTruthEntries.length > 0 ? "text-blue-600" : "text-slate-400"}`}>
                  {groundTruthEntries.length > 0 ? `Loaded (${groundTruthEntries.length})` : "Missing"}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-500 font-medium">Training Log:</span>
                <span className={`font-bold text-right ${trainingLogEntries.length > 0 ? "text-indigo-600" : "text-slate-400"}`}>
                  {trainingLogEntries.length > 0 ? `${trainingLogEntries.length} logs` : "Missing"}
                </span>
              </div>
            </div>
          </div>

          {/* DATASET DIRECTORY MANAGER */}
          {setSelectedDir && availableDirs.length > 0 && (
            <div className="space-y-1.5 bg-[#f8fafc] p-3 rounded-xl border border-slate-200/85">
              <div className="flex items-center justify-between">
                <label className="block text-[10px] font-mono font-bold text-slate-500 uppercase tracking-wide">
                  DATASET WORKSPACE
                </label>
                <span className="bg-indigo-50 text-indigo-700 text-[8px] px-1.5 py-0.5 rounded-sm font-mono font-bold border border-indigo-200 shrink-0">
                  Active Folder
                </span>
              </div>
              <select
                value={selectedDir}
                onChange={(e) => setSelectedDir(e.target.value)}
                className="w-full bg-white border border-slate-200 text-slate-700 text-xs rounded-lg px-2.5 py-1.5 font-semibold focus:outline-hidden focus:border-indigo-500 transition shadow-3xs font-mono cursor-pointer"
              >
                {availableDirs.map((name) => (
                  <option key={name} value={name}>
                    {name === "Default Workspace" ? "📁 Default Workspace" : `📁 ${name}`}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* MEDIA ASSETS box */}
          <div className="space-y-1">
            <h4 className="text-[10px] font-mono font-black text-slate-400 uppercase tracking-wider">MEDIA ASSETS ({files.filter((f) => f.type === "image").length})</h4>
            {files.filter((f) => f.type === "image").length === 0 ? (
              <div className="bg-slate-55 border border-slate-200/80 rounded-lg py-3 text-center text-slate-400 font-mono text-[9px] uppercase font-bold tracking-wider">
                NO IMAGES CACHED
              </div>
            ) : (
              <div className="max-h-[140px] overflow-y-auto space-y-1 border border-slate-200/60 p-2 rounded-lg bg-slate-50/50 hover:bg-slate-50 transition custom-scrollbar text-[10.5px] font-mono">
                {files
                  .filter((f) => f.type === "image")
                  .map((img) => {
                    const trialBase = getTrialBase(img.name);
                    const angle = normalizeAngle(extractAngle(img.name));
                    const imgKey = `${trialBase}::${angle}`;
                    const isSelected = selectedResultKey === imgKey;

                    return (
                      <button
                        key={img.name}
                        onClick={() => setSelectedResultKey(imgKey)}
                        className={`w-full flex items-center justify-between gap-2 border-b border-dashed border-slate-100 pb-1 last:border-0 last:pb-0 text-left px-1.5 py-1 rounded transition-all cursor-pointer select-none ${
                          isSelected 
                            ? "bg-slate-900 text-white font-extrabold shadow-sm" 
                            : "text-slate-600 hover:bg-slate-100 font-medium"
                        }`}
                        title={`Click to preview image: ${img.name}`}
                      >
                        <span className="truncate flex-1 font-semibold">{img.name}</span>
                        <span className={`text-[9px] shrink-0 font-medium ${isSelected ? "text-slate-300" : "text-slate-400"}`}>
                          {(img.size / 1024).toFixed(0)} KB
                        </span>
                      </button>
                    );
                  })}
              </div>
            )}
          </div>

          {/* UPDATE DATA FEED section */}
          <div className="space-y-3.5 pt-1">
            <h4 className="text-[10px] font-mono font-black text-slate-400 uppercase tracking-wider border-b border-slate-100 pb-1.5">UPDATE DATA FEED</h4>

            {/* TRAINING LOG UPLOAD */}
            <div className="space-y-1">
              <label className="block text-[10px] font-mono font-bold text-slate-500 uppercase">TRAINING LOG</label>
              <label className="flex items-center justify-between border border-slate-200 rounded-lg px-2.5 py-1.5 bg-white hover:bg-slate-50 transition cursor-pointer text-[11px]">
                <span className="text-slate-600 font-medium font-mono">Upload CSV</span>
                <Upload className="w-3.5 h-3.5 text-slate-400" />
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleTrainingLogUpload}
                  className="hidden"
                />
              </label>
            </div>

            {/* RESULTS UPLOAD & LOAD FROM LOCAL STORAGE */}
            <div className="space-y-2 bg-[#f8fafc] p-3 rounded-xl border border-slate-200/85">
              <div className="flex items-center justify-between">
                <label className="block text-[10px] font-mono font-bold text-slate-500 uppercase tracking-wide">RESULTS (PREDICTIONS)</label>
                {predictionEntries.length > 0 && (
                  <span className="bg-emerald-50 text-emerald-700 text-[8px] px-1.5 py-0.5 rounded-sm font-mono font-bold border border-emerald-200 shrink-0">
                     Active
                  </span>
                )}
              </div>

              {/* ACTIVE FILENAME DISPLAYER */}
              {predictionEntries.length > 0 && (
                <div className="bg-white border border-indigo-100/80 rounded-lg p-2 flex items-center justify-between text-[10px] font-mono shadow-3xs">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <FileSpreadsheet className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
                    <span className="text-slate-700 font-bold truncate max-w-[160px]" title={predictionsFilename || "results-yolo8m-obb"}>
                      {predictionsFilename || "results-yolo8m-obb"}
                    </span>
                  </div>
                  <span className="text-slate-400 shrink-0 font-medium">({predictionEntries.length} pts)</span>
                </div>
              )}

              <div className="space-y-2 pt-0.5">
                {/* Option 1: Load from server storage */}
                {csvOutputs.length > 0 && (
                  <div className="space-y-1">
                    <span className="block text-[8.5px] font-mono font-bold text-slate-400 uppercase tracking-wider">Load Server Runs:</span>
                    <div className="flex gap-1.5 items-center w-full">
                      <select
                        className="flex-1 min-w-0 bg-white border border-slate-200 rounded px-1.5 py-1 text-[10px] font-mono text-slate-700 focus:outline-indigo-500 shadow-3xs"
                        value={selectedServerCsv}
                        onChange={(e) => setSelectedServerCsv(e.target.value)}
                      >
                        <option value="">- Select Run -</option>
                        {csvOutputs.map((csv) => {
                          const displayName = csv.filename.includes("/") 
                            ? csv.filename.split("/").pop() 
                            : csv.filename.includes("\\") 
                              ? csv.filename.split("\\").pop() 
                              : csv.filename;
                          return (
                            <option key={csv.filename} value={csv.filename}>
                              {displayName}
                            </option>
                          );
                        })}
                      </select>
                      <button
                        onClick={handleLoadSelectedServerCsv}
                        disabled={!selectedServerCsv}
                        className="flex-shrink-0 px-2.5 py-1 bg-slate-800 disabled:opacity-50 text-white hover:bg-slate-900 rounded text-[10.5px] font-mono font-bold transition cursor-pointer shadow-3xs min-w-[48px]"
                      >
                        Load
                      </button>
                    </div>
                  </div>
                )}

                {/* Option 2: Upload local file */}
                <div className="space-y-1">
                  <span className="block text-[8.5px] font-mono font-bold text-slate-400 uppercase tracking-wider">
                     {csvOutputs.length > 0 ? "Or Upload Custom .csv:" : "Upload Custom .csv:"}
                  </span>
                  <label className="flex items-center justify-between border border-slate-200 rounded-lg px-2.5 py-1.5 bg-white hover:bg-slate-50 transition cursor-pointer text-[10.5px] font-mono shadow-3xs">
                    <span className="text-slate-500 font-medium truncate max-w-[190px]">
                      Choose local file...
                    </span>
                    <Upload className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                    <input
                      type="file"
                      accept=".csv"
                      onChange={handlePredictionsUpload}
                      className="hidden"
                    />
                  </label>
                </div>
              </div>
            </div>

            {/* GROUND TRUTH BENCHMARK UPLOAD */}
            <div className="space-y-1">
              <label className="block text-[10px] font-mono font-bold text-slate-500 uppercase">GROUND TRUTH (TARGET)</label>
              <label className="flex items-center justify-between border border-slate-200 rounded-lg px-2.5 py-1.5 bg-white hover:bg-slate-50 transition cursor-pointer text-[11px]">
                <span className="text-slate-600 font-medium font-mono">Upload CSV</span>
                <Upload className="w-3.5 h-3.5 text-slate-400" />
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleGroundTruthUpload}
                  className="hidden"
                />
              </label>
            </div>

          </div>

        </div>

        {/* FIXED BOTTOM SIDENEWS */}
        <div className="p-4 border-t border-slate-100 bg-slate-50">
          <button
            onClick={handleTriggerExportPredictions}
            disabled={predictionEntries.length === 0}
            className="w-full py-2 bg-[#0f172a] hover:bg-[#1e293b] disabled:bg-slate-200 disabled:text-slate-400 disabled:cursor-not-allowed text-white rounded-lg text-xs font-bold font-mono transition shadow-xs cursor-pointer flex items-center justify-center gap-1.5"
          >
            <Download className="w-3.5 h-3.5" />
            Export Predictions (.csv)
          </button>
        </div>

      </div>

      {/* 2. MIDDLE COLUMN: FILE LIST & SEARCH FILTER */}
      <div className="w-[330px] border-r border-slate-200 bg-white h-full flex flex-col flex-shrink-0 select-none">
        
        {/* Title / Info block */}
        <div className="p-4 border-b border-slate-100 flex items-center justify-between">
          <h3 className="text-xs font-mono font-black text-slate-500 uppercase tracking-wider">
            FILE LIST ({filteredReconciledResults.length})
          </h3>
          <span className="text-[9.5px] font-mono font-bold px-1.5 py-0.5 bg-slate-100 text-slate-600 rounded">
            TOTAL: {reconciledResults.length}
          </span>
        </div>

        {/* Search bar & filter toggle block */}
        <div className="p-3 border-b border-slate-100 space-y-3 bg-slate-50/50">
          
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-2.5" />
            <input
              type="text"
              placeholder="Search file base..."
              value={fileFilter}
              onChange={(e) => setFileFilter(e.target.value)}
              className="w-full bg-white border border-slate-200 rounded-lg pl-8 pr-3 py-1.5 text-xs font-mono focus:outline-none focus:border-indigo-500"
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label className="flex items-center justify-between gap-1.5 text-[11px] font-mono text-slate-650 cursor-pointer select-none">
              <span className="flex items-center gap-1.5">
                <input
                  type="checkbox"
                  checked={lowAccuracyFilter}
                  onChange={(e) => setLowAccuracyFilter(e.target.checked)}
                  className="accent-red-650 w-3.5 h-3.5 rounded"
                />
                <span className="font-bold text-slate-700">Low Accuracy Filter</span>
              </span>
              <span className="bg-red-50 text-red-600 font-extrabold border border-red-200 text-[9px] px-1.5 rounded-full items-center">
                &gt; {lowAccuracyThreshold} px
              </span>
            </label>
            {lowAccuracyFilter && (
              <input
                type="range"
                min="1"
                max="25"
                step="0.5"
                value={lowAccuracyThreshold}
                onChange={(e) => setLowAccuracyThreshold(Number(e.target.value))}
                className="w-full accent-red-600 cursor-pointer h-1 bg-red-100 rounded"
              />
            )}
          </div>

        </div>

        {/* DATASET SUMMARY */}
        <div className="px-4 py-3 border-b border-slate-100 bg-slate-50/30 grid grid-cols-2 gap-2 text-center">
          <div className="p-2 bg-white rounded-lg border border-slate-200/80">
            <div className="text-[9px] font-mono font-black text-slate-400 uppercase leading-none">AVG ERROR</div>
            <div className={`text-sm font-black mt-1 font-sans ${stats.avgErr !== null && stats.avgErr > 5 ? "text-amber-600" : "text-slate-800"}`}>
              {stats.avgErr !== null ? `${stats.avgErr.toFixed(2)}px` : "0.00px"}
            </div>
          </div>
          <div className="p-2 bg-white rounded-lg border border-slate-200/80">
            <div className="text-[9px] font-mono font-black text-slate-400 uppercase leading-none">MAX ERROR</div>
            <div className={`text-sm font-black mt-1 font-sans ${stats.maxErr > 10 ? "text-red-600 font-extrabold" : "text-slate-800"}`}>
              {stats.maxErr !== 0 ? `${stats.maxErr.toFixed(2)}px` : "0.00px"}
            </div>
          </div>
        </div>

        {/* Main list segment */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1 custom-scrollbar">
          {filteredReconciledResults.length === 0 ? (
            <div className="text-center py-12 text-slate-400 flex flex-col items-center justify-center p-4">
              <Info className="w-5 h-5 text-slate-300 mb-2" />
              <p className="text-[11px] font-mono leading-relaxed">No matching calibrated records found.</p>
            </div>
          ) : (
            filteredReconciledResults.map((result) => {
              const isSelected = selectedResultKey === result.key;
              const isOverridden = !!modifiedPredictions[result.key];
              const nodeLabel = `NODE_S${result.angle}`;
              const trialBase = getTrialBase(result.filename);

              return (
                <button
                  key={result.key}
                  onClick={() => setSelectedResultKey(result.key)}
                  className={`w-full text-left p-2.5 rounded-lg transition-all border flex flex-col gap-1 cursor-pointer ${
                    isSelected
                      ? "bg-slate-900 border-slate-950 text-white shadow-md shadow-slate-950/20"
                      : "bg-white hover:bg-slate-50 border-slate-100/80 text-slate-700 hover:border-slate-300"
                  }`}
                >
                  <div className="flex items-center justify-between font-mono text-[10px] w-full gap-2">
                    <span className={`font-black font-mono tracking-wider px-1.5 py-0.5 rounded text-[8.5px] uppercase ${isSelected ? "bg-indigo-650 text-white font-black" : "bg-indigo-50 text-indigo-700 font-bold"}`}>
                      {nodeLabel}
                    </span>
                    {isOverridden && (
                      <span className="bg-amber-100 text-amber-700 border border-amber-250 font-black text-[8px] uppercase px-1 rounded">
                        STAGED
                      </span>
                    )}
                  </div>
                  
                  <div className="text-[11px] font-mono font-medium leading-tight truncate w-full mt-0.5">
                    {trialBase}
                  </div>

                  <div className="flex gap-1 mt-1">
                    {result.prediction && (
                      <span className={`text-[8.5px] px-1 py-0.2 rounded-sm font-bold ${
                        isSelected ? "bg-slate-800 text-slate-200" : "bg-emerald-50 text-emerald-700 border border-emerald-100"
                      }`}>
                        PRED
                      </span>
                    )}
                    {result.groundTruth && (
                      <span className={`text-[8.5px] px-1 py-0.2 rounded-sm font-bold ${
                        isSelected ? "bg-slate-800 text-slate-200" : "bg-blue-50 text-blue-700 border border-blue-100"
                      }`}>
                        GT
                      </span>
                    )}
                    {result.imageUrl && (
                      <span className="text-[8.5px] px-1 py-0.1 border border-teal-100 bg-teal-50 text-teal-700 rounded-sm font-mono font-medium">
                        IMG
                      </span>
                    )}
                  </div>
                </button>
              );
            })
          )}
        </div>

      </div>

      {/* 3. CENTRAL WORKSPACE: INTERACTIVE VISUAL INSPECTOR (Radar canvas) */}
      <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-5 select-none h-full custom-scrollbar">
        
        {/* Top visual warning/success broadcast */}
        {panelMessage && (
          <div className={`p-4 rounded-xl text-xs flex items-center gap-3 shadow-3xs animate-fade-in ${
            panelMessage.type === "success" 
              ? "bg-emerald-50 border border-emerald-200 text-emerald-800" 
              : "bg-red-50 border border-red-200 text-red-800"
          }`}>
            {panelMessage.type === "success" ? <Check className="w-4 h-4 text-emerald-600" /> : <AlertTriangle className="w-4 h-4 text-red-600" />}
            <span className="flex-1 font-mono font-medium">{panelMessage.text}</span>
            <button onClick={() => setPanelMessage(null)} className="opacity-60 hover:opacity-100 font-black px-1">✕</button>
          </div>
        )}

        {/* Selected Image Title & Quick Jump buttons */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex items-center justify-between gap-4">
          <div className="truncate flex-1">
            <div className="flex flex-wrap items-center gap-3">
              <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2 truncate">
                <span className="inline-block w-2.5 h-2.5 rounded-full bg-indigo-505 shrink-0 animate-pulse"></span>
                {selectedResult ? getTrialBase(selectedResult.filename) : "None Selected"}
              </h2>
              {selectedResult && (
                <div className="flex items-center gap-1.5 shrink-0">
                  <span className="inline-flex items-center gap-1 bg-indigo-50 hover:bg-indigo-100/80 border border-indigo-100 text-indigo-700 text-[10.5px] font-mono font-bold px-2 py-0.5 rounded transition shadow-3xs">
                    {getTrialBase(selectedResult.filename)}.mat
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(`${getTrialBase(selectedResult.filename)}.mat`);
                        setPanelMessage({ type: "success", text: `Permanently copied filename '${getTrialBase(selectedResult.filename)}.mat' to clipboard!` });
                      }}
                      className="hover:text-indigo-900 focus:outline-none cursor-pointer ml-1 text-indigo-400"
                      title="Copy .mat filename to clipboard"
                    >
                      <Copy className="w-3 h-3" />
                    </button>
                  </span>
                </div>
              )}
            </div>
            <div className="text-[10px] font-mono text-slate-500 mt-1.5 truncate max-w-[500px]">
              Reference Resource Suffix: {selectedResult ? selectedResult.filename : "No dataset loaded"}
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            {getBadgeTag()}

            <span className="h-6 w-px bg-slate-200"></span>

            <div className="flex bg-slate-100 rounded-lg p-0.5 border border-slate-200">
              <button
                onClick={handleSelectedResultPrev}
                disabled={filteredReconciledResults.length === 0}
                className="p-1 px-1.5 hover:bg-white text-slate-600 disabled:opacity-30 rounded-md transition cursor-pointer"
                title="Previous Image (Hold Shift to jump fast)"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <button
                onClick={handleSelectedResultNext}
                disabled={filteredReconciledResults.length === 0}
                className="p-1 px-1.5 hover:bg-white text-slate-600 disabled:opacity-30 rounded-md transition cursor-pointer"
                title="Next Image"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* ACTIVE NODE AND STATUS ROW */}
        {selectedResult && (
          <div className="flex items-center justify-between bg-white border border-slate-200 rounded-xl px-4 py-2 text-xs font-mono shadow-3xs">
            <div className="flex items-center gap-2">
              <span className="text-slate-400 font-black">ACTIVE NODE:</span>
              <span className="font-extrabold text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200/50">
                S{selectedResult.angle}
              </span>
            </div>
            {displacementError !== null && (
              <div className="flex items-center gap-1">
                <span className="text-slate-400 font-black">DISPLACEMENT L2:</span>
                <span className={`font-black tracking-tight ${displacementError > 5.0 ? "text-amber-600 font-extrabold animate-pulse" : "text-emerald-600"}`}>
                  {displacementError.toFixed(3)} px
                </span>
              </div>
            )}
          </div>
        )}

        {/* WORK BENCH VIEW - Dark Canvas with custom Legend overlay */}
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm flex flex-col gap-3">
          
          <div className="flex items-center justify-between border-b border-slate-150 pb-2 flex-wrap gap-2 text-xs">
            <span className="font-mono font-black text-slate-400 tracking-wider">VISUAL INSPECTOR</span>
            <div className="flex items-center gap-3 font-mono font-bold text-[10.5px]">
              <span className="flex items-center gap-1 text-emerald-600">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block"></span>
                ● PREDICTED
              </span>
              <span className="flex items-center gap-1 text-blue-600">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block"></span>
                ● GROUND TRUTH
              </span>
            </div>
          </div>

          <div className="flex justify-center bg-[#090d16] p-4 rounded-xl border border-slate-900">
            <div className="w-full max-w-[640px] aspect-[4/3] relative rounded-lg overflow-hidden shadow-2xl">
              <svg
                ref={containerRef}
                onPointerDown={handleSvgPointerDown}
                onPointerMove={handlePointerMove}
                onPointerUp={handlePointerUp}
                onWheel={handleWheel}
                onDoubleClick={handleDoubleClick}
                width="100%"
                height="100%"
                viewBox="0 0 640 480"
                className="bg-[#0e1624] block select-none relative cursor-crosshair overflow-hidden"
              >
                {/* SVG RENDERING CONTAINER TRANSLATED AND ZOOMED */}
                <g transform={`translate(${offset.x}, ${offset.y}) scale(${zoom})`}>

                  {/* SVG RENDERING BACKGROUND IMAGE IF EXISTS */}
                  {selectedResult?.imageUrl ? (
                    <image
                      href={selectedResult.imageUrl}
                      width="640"
                      height="480"
                      referrerPolicy="no-referrer"
                      className="opacity-95 select-none pointer-events-none"
                    />
                  ) : (
                    <>
                      {/* SVG target sweep watermark radar */}
                      <g opacity="0.14">
                        <circle cx="320" cy="240" r="150" fill="none" stroke="#3b82f6" strokeWidth="1" strokeDasharray="4 4" />
                        <circle cx="320" cy="240" r="100" fill="none" stroke="#3b82f6" strokeWidth="1" />
                        <circle cx="320" cy="240" r="50" fill="none" stroke="#3b82f6" strokeWidth="1.5" />
                        <line x1="120" y1="240" x2="520" y2="240" stroke="#3b82f6" strokeWidth="1" strokeDasharray="2 2" />
                        <line x1="320" y1="40" x2="320" y2="440" stroke="#3b82f6" strokeWidth="1" strokeDasharray="2 2" />
                        
                        {/* Compass grid lines */}
                        <line x1="178" y1="98" x2="462" y2="382" stroke="#3b82f6" strokeWidth="0.5" strokeDasharray="2 2" />
                        <line x1="178" y1="382" x2="462" y2="98" stroke="#3b82f6" strokeWidth="0.5" strokeDasharray="2 2" />
                      </g>

                      <text x="325" y="30" fill="#3b82f6" fontSize="10" fontFamily="monospace" opacity="0.3">Y=0 CANVAS MINIMUM</text>
                      <text x="325" y="470" fill="#3b82f6" fontSize="10" fontFamily="monospace" opacity="0.3">Y=480 CANVAS BOUNDS</text>
                      <text x="10" y="235" fill="#3b82f6" fontSize="10" fontFamily="monospace" opacity="0.3">X=0 MIN</text>
                      <text x="545" y="235" fill="#3b82f6" fontSize="10" fontFamily="monospace" opacity="0.3">X=640 MAX</text>

                      {/* Simulation blueprint text watermark */}
                      <text x="320" y="230" fill="#94a3b8" fontSize="11" fontFamily="sans-serif" textAnchor="middle" opacity="0.15">
                        PSEUDOPUPIL PROCEDURAL CALIBRATION CANVAS
                      </text>
                      <text x="320" y="250" fill="#94a3b8" fontSize="9" fontFamily="monospace" textAnchor="middle" opacity="0.12">
                        No matching high-res JPG loaded in visible/ directory
                      </text>
                    </>
                  )}

                  {/* COORDINATE GRID LINE OVERLAYS UNCONDITIONALLY SHOWN IF SELECTED */}
                  {showGrid && (
                    <g opacity="0.12" stroke="#64748b" strokeWidth="0.5">
                      {Array.from({ length: 12 }).map((_, idx) => {
                        const x = (idx + 1) * 50;
                        return <line key={`x-${idx}`} x1={x} y1="0" x2={x} y2="480" />;
                      })}
                      {Array.from({ length: 9 }).map((_, idx) => {
                        const y = (idx + 1) * 50;
                        return <line key={`y-${idx}`} x1="0" y1={y} x2="640" y2={y} />;
                      })}
                    </g>
                  )}

                  {/* TARGET REFERENCE BENCHMARKS (GROUND TRUTH) */}
                  {showGroundTruth && gtX !== null && gtY !== null && (
                    <g>
                      {(() => {
                        const { x, y } = getRenderCoords(gtX, gtY);
                        return (
                          <>
                            {/* Deep blue target dot with customizable radius */}
                            <circle cx={x} cy={y} r={plotDotRadius} fill="#2563eb" />
                          </>
                        );
                      })()}
                    </g>
                  )}

                  {/* REFINED PREDICTION DRAGGABLE POINT (Green hex #10b981) */}
                  {showPredictions && fineX !== null && fineY !== null && (
                    <g>
                      {(() => {
                        const { x, y } = getRenderCoords(fineX, fineY);
                        return (
                          <>
                            {/* Transparent interactive target grab helper */}
                            <circle
                              cx={x}
                              cy={y}
                              r={Math.max(15, plotDotRadius * 2.5)}
                              fill="transparent"
                              className="cursor-move"
                              onPointerDown={(e) => handlePointerDown(e, "fine")}
                            />
                            {/* Green dot with customizable radius */}
                            <circle cx={x} cy={y} r={plotDotRadius} fill="#10b981" className="pointer-events-none" />
                          </>
                        );
                      })()}
                    </g>
                  )}

                </g>
              </svg>
            </div>
          </div>

          {/* Status footer for coordinates info inside Visual Workspace container */}
          <div className="bg-slate-900 border border-slate-950 p-3 rounded-xl flex items-center justify-between font-mono text-[10.5px] text-slate-400">
            <div className="flex gap-4">
              <span>SOURCE: <strong className="text-slate-200">{naturalSize.width}X{naturalSize.height}</strong></span>
              <span>CURSOR: <strong className="text-slate-200">{fineX !== null ? `${Math.round(fineX)}, ${Math.round(fineY ?? 0)}` : "DEFAULT (COARSE)"}</strong></span>
              <span>ZOOM: <strong className="text-indigo-400 font-bold">{zoom.toFixed(2)}X</strong></span>
            </div>
            <div className="flex gap-2.5 items-center font-bold">
              <span className="text-emerald-500">● PREDICTED</span>
              <span className="text-blue-500">● TRUTH</span>
            </div>
          </div>

        </div>

        {/* RECHARTS LOSS CURVES VISUALIZATION (Dynamic loss graph) */}
        {trainingLogEntries.length > 0 && (
          <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2.5 mb-4 flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4.5 h-4.5 text-indigo-600" />
                <h4 className="text-xs font-bold font-mono text-slate-500 uppercase">Loss Curves & Trends</h4>
              </div>
              <span className="text-[10px] font-mono bg-indigo-50 text-indigo-600 px-2.5 py-0.5 rounded-full font-extrabold border border-indigo-200/50">
                {trainingLogEntries.length} Epochs loaded
              </span>
            </div>

            <div className="w-full h-[240px] font-mono text-xs">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={trainingLogEntries}
                  margin={{ top: 5, right: 10, left: -25, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="epoch" tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 9 }} />
                  <Tooltip wrapperStyle={{ fontFamily: "monospace", fontSize: "10px" }} />
                  <Legend wrapperStyle={{ fontSize: "10.5px" }} />
                  <Line
                    type="monotone"
                    dataKey="train_loss"
                    stroke="#10b981"
                    strokeWidth={2}
                    activeDot={{ r: 5 }}
                    name="Train Loss"
                  />
                  <Line
                    type="monotone"
                    dataKey="val_loss"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    activeDot={{ r: 5 }}
                    name="Val Loss"
                  />
                  <Line
                    type="monotone"
                    dataKey="lr"
                    stroke="#f59e0b"
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    name="LR"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

      </div>

      {/* 4. RIGHT COLUMN: COORDINATES EDITORS & STATUS CONTROLS */}
      <div className="w-[300px] border-l border-slate-200 bg-[#fefefe] h-full flex flex-col flex-shrink-0 p-5 overflow-y-auto custom-scrollbar gap-5">
        
        {/* COORDINATE INSPECTOR CARD */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <div className="text-[10px] font-mono text-indigo-600 font-bold tracking-wider mb-1 uppercase">COORDINATE INSPECTOR</div>
          <p className="text-[10.5px] text-slate-400 font-mono mb-4 uppercase">PREDICTED CENTROID (INTERACTIVE)</p>
          
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-[10px] font-mono text-slate-400 font-bold uppercase mb-1">CX</label>
              <input
                type="number"
                step="0.01"
                value={fineX !== null ? Number(fineX.toFixed(2)) : ""}
                onChange={(e) => {
                  if (!selectedResult) return;
                  const val = Number(e.target.value);
                  setModifiedPredictions(prev => ({
                    ...prev,
                    [selectedResult.key]: { cx: val, cy: fineY ?? 0 }
                  }));
                }}
                className="w-full bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 font-mono text-xs text-slate-800 font-bold focus:outline-none focus:border-indigo-500"
              />
            </div>
            
            <div>
              <label className="block text-[10px] font-mono text-slate-400 font-bold uppercase mb-1">CY</label>
              <input
                type="number"
                step="0.01"
                value={fineY !== null ? Number(fineY.toFixed(2)) : ""}
                onChange={(e) => {
                  if (!selectedResult) return;
                  const val = Number(e.target.value);
                  setModifiedPredictions(prev => ({
                    ...prev,
                    [selectedResult.key]: { cx: fineX ?? 0, cy: val }
                  }));
                }}
                className="w-full bg-slate-50 border border-slate-200 rounded px-2.5 py-1.5 font-mono text-xs text-slate-800 font-bold focus:outline-none focus:border-indigo-500"
              />
            </div>
          </div>

          <div className="mt-4 pt-1">
            <label className="block text-[9px] font-mono text-slate-400 font-black uppercase tracking-wider mb-1">ERROR (L2 DISPLACEMENT)</label>
            <div className="text-3xl font-black text-slate-900 leading-none flex items-baseline gap-1 font-sans">
              <span>{displacementError !== null ? displacementError.toFixed(2) : "0.00"}</span>
              <span className="text-[11px] font-mono font-bold text-slate-400 uppercase">px</span>
            </div>
          </div>

          {/* Block functional buttons */}
          <div className="mt-5 space-y-2">
            <button
              onClick={handleOverrideAndSave}
              disabled={!selectedResult}
              className="w-full py-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-45 text-white rounded-lg text-xs font-bold transition flex items-center justify-center gap-1 cursor-pointer"
            >
              <Check className="w-3.5 h-3.5" />
              Override & Save Label
            </button>
            
            <button
              onClick={handleResetActiveOverride}
              disabled={!selectedResult || !modifiedPredictions[selectedResult.key]}
              className="w-full py-2 border border-slate-200 text-slate-600 hover:text-slate-800 bg-white hover:bg-slate-50 rounded-lg text-xs font-semibold transition cursor-pointer"
            >
              Revert Changes
            </button>
          </div>
        </div>

        {/* CONFIDENCE SCORE CARD */}
        <div className="bg-gradient-to-br from-[#0f172a] to-[#1e293b] text-white border border-slate-950 rounded-xl p-5 shadow-md relative overflow-hidden">
          <div className="text-[10px] font-mono text-slate-400 font-extrabold uppercase tracking-wider mb-1.5">CONFIDENCE SCORE</div>
          <div className="text-2xl font-black text-white flex items-baseline gap-1 font-sans">
            <span>{selectedResult?.prediction?.confidence ? selectedResult.prediction.confidence.toFixed(4) : "0.0000"}</span>
            <span className="text-[10px] font-mono text-slate-550 uppercase">CALCULATED</span>
          </div>
          
          <div className="absolute right-3.5 bottom-3 text-slate-800 opacity-25 font-black text-5xl pointer-events-none select-none">
            ⓘ
          </div>
        </div>

        {/* GLOBAL CALIBRATION WINDOW */}
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <div className="text-[10px] font-mono text-slate-400 font-bold uppercase tracking-wider mb-4">GLOBAL CALIBRATION</div>
          
          <div className="space-y-4">
            
            {/* Dot Radius Slider */}
            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-xs font-mono">
                <span className="text-slate-500">Plot Dot Radius: <span className="text-emerald-600 font-extrabold">{plotDotRadius}px</span></span>
                <button
                  onClick={() => setPlotDotRadius(3.0)}
                  className="text-emerald-600 hover:underline text-[10px] font-bold"
                >
                  Reset
                </button>
              </div>
              <input
                type="range"
                min="1"
                max="25"
                step="0.5"
                value={plotDotRadius}
                onChange={(e) => setPlotDotRadius(Number(e.target.value))}
                className="w-full h-1 bg-slate-100 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
            </div>

            {/* Display Calibration Toggles */}
            <div className="pt-2 border-t border-slate-100 space-y-2">
              <label className="flex items-center gap-2 text-[11px] text-slate-600 hover:text-slate-800 cursor-pointer select-none font-mono">
                <input
                  type="checkbox"
                  checked={showPredictions}
                  onChange={(e) => setShowPredictions(e.target.checked)}
                  className="accent-emerald-600 w-3.5 h-3.5 rounded border-slate-300"
                />
                <span className="font-semibold text-emerald-700">Show Predictions</span>
              </label>

              <label className="flex items-center gap-2 text-[11px] text-slate-600 hover:text-slate-800 cursor-pointer select-none font-mono">
                <input
                  type="checkbox"
                  checked={showGroundTruth}
                  onChange={(e) => setShowGroundTruth(e.target.checked)}
                  className="accent-blue-600 w-3.5 h-3.5 rounded border-slate-300"
                />
                <span className="font-semibold text-blue-700">Show Ground Truth</span>
              </label>

              <label className="flex items-center gap-2 text-[11px] text-slate-600 hover:text-slate-800 cursor-pointer select-none font-mono">
                <input
                  type="checkbox"
                  checked={showGrid}
                  onChange={(e) => setShowGrid(e.target.checked)}
                  className="accent-indigo-600 w-3.5 h-3.5 rounded border-slate-300"
                />
                <span className="font-semibold">Show Custom Grid</span>
              </label>
            </div>

          </div>
        </div>

        {/* Quick Tips */}
        <div className="bg-blue-50/50 border border-blue-150 rounded-xl p-4 text-[10.5px] text-indigo-900 leading-relaxed font-mono">
          <div className="font-bold flex items-center gap-1 mb-1">
            <Sliders className="w-3.5 h-3.5 font-bold text-indigo-500" />
            STATION OVERRIDES
          </div>
          Drag points on the screen viewport. Overrides are staged locally and stored dynamically until compiled output export.
        </div>

      </div>

    </div>
  );
}
