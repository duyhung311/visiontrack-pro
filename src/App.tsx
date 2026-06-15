import React, { useState, useEffect, useRef } from "react";
import {
  FileCode,
  FolderOpen,
  Settings,
  Brain,
  Cpu,
  Play,
  Square,
  ChevronRight,
  Download,
  Trash,
  Info,
  AlertTriangle,
  CheckCircle,
  Clock,
  RefreshCw,
  Terminal,
  Database,
  ArrowRight,
  Layers,
  FileSpreadsheet,
  Save,
  Undo,
  Eye,
  EyeOff,
  Upload,
  Check
} from "lucide-react";
import {
  FileInfo,
  TrainingStatus,
  InferenceStatus,
  RunHistoryItem,
  CSVOutputItem,
  TrainingLogEntry,
  PredictionEntry,
  GroundTruthEntry,
  ImageResult
} from "./types";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";
import {
  parseTrainingLogs,
  parsePredictions,
  parseGroundTruth,
  reconcileDataset,
  exportPredictionsCSV,
  exportGroundTruthCSV,
  getTrialBase,
  extractAngle,
  normalizeAngle
} from "./lib/csvParser";
import VisionTrackInspectorPanel from "./components/VisionTrackInspectorPanel";

export default function App() {
  // Navigation tabs - simple tab selector inside a bento container
  const [activeTab, setActiveTab] = useState<"dashboard" | "inspector" | "logs">("dashboard");

  // VisionTrack Pro States
  const [trainingLogCsvText, setTrainingLogCsvText] = useState("");
  const [predictionsCsvText, setPredictionsCsvText] = useState("");
  const [predictionsFilename, setPredictionsFilename] = useState("");
  const [groundTruthCsvText, setGroundTruthCsvText] = useState("");

  const [trainingLogEntries, setTrainingLogEntries] = useState<TrainingLogEntry[]>([]);
  const [predictionEntries, setPredictionEntries] = useState<PredictionEntry[]>([]);
  const [groundTruthEntries, setGroundTruthEntries] = useState<GroundTruthEntry[]>([]);

  const [modifiedPredictions, setModifiedPredictions] = useState<Record<string, { cx: number; cy: number }>>({});
  const [modifiedCoarse, setModifiedCoarse] = useState<Record<string, { cx: number; cy: number }>>({});

  const [selectedResultKey, setSelectedResultKey] = useState<string | null>(null);

  // Calibration and Display Configs
  const [invertY, setInvertY] = useState(false);
  const [yOffset, setYOffset] = useState(0);
  const [showCoarse, setShowCoarse] = useState(true);
  const [showGroundTruth, setShowGroundTruth] = useState(true);
  const [showPredictions, setShowPredictions] = useState(true);
  const [showGrid, setShowGrid] = useState(true);

  // Svg pointer dragging state
  const [activeDrag, setActiveDrag] = useState<{ pointType: "fine" | "coarse" } | null>(null);

  // State Stores
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [localImageFiles, setLocalImageFiles] = useState<{ name: string; size: number; url: string }[]>([]);
  const [runs, setRuns] = useState<RunHistoryItem[]>([]);
  const [csvOutputs, setCsvOutputs] = useState<CSVOutputItem[]>([]);
  const [systemStatus, setSystemStatus] = useState<{
    training: TrainingStatus;
    inference: InferenceStatus;
  } | null>(null);

  // File explorer states
  const [localSourcePath, setLocalSourcePath] = useState("");
  const [localScanResult, setLocalScanResult] = useState<{
    message: string;
    copiedCount: number;
    copiedFiles?: string[];
  } | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);

  // Training inputs configuration state
  const [trainConfig, setTrainConfig] = useState({
    version: "8", // 8, 11, 26 only
    size: "m",   // n, s, m, l only
    epochs: 10,
    batch: 16,
    device: "cpu",
    labelFile: "",
  });

  // Inference inputs configuration state
  const [inferConfig, setInferConfig] = useState({
    version: "8",
    size: "m",
    device: "cpu",
  });

  // Conda environments states
  const [condaEnvs, setCondaEnvs] = useState<string[]>([]);
  const [selectedCondaEnv, setSelectedCondaEnv] = useState<string>("system-default");
  const [customCondaEnv, setCustomCondaEnv] = useState<string>("");
  const [isCondaLoading, setIsCondaLoading] = useState<boolean>(false);
  const [condaInstalled, setCondaInstalled] = useState<boolean>(true);

  // Dynamic visible directories state
  const [availableDirs, setAvailableDirs] = useState<string[]>(["Default Workspace"]);
  const [selectedDir, setSelectedDir] = useState<string>("Default Workspace");
  const [selectedTrainDir, setSelectedTrainDir] = useState<string>("Default Workspace");
  const [selectedInferDir, setSelectedInferDir] = useState<string>("Default Workspace");

  const [trainFiles, setTrainFiles] = useState<any[]>([]);
  const [inferFiles, setInferFiles] = useState<any[]>([]);

  // Merge server files with local object URLs for image entries
  const mergedFiles = React.useMemo(() => {
    const fileMap = new Map<string, FileInfo>();
    
    // 1. Add server files
    files.forEach((file) => {
      fileMap.set(file.name, file);
    });

    // 2. Add local files
    localImageFiles.forEach((file) => {
      fileMap.set(file.name, {
        name: file.name,
        size: file.size,
        mtime: new Date().toISOString(),
        type: "image",
      });
    });

    return Array.from(fileMap.values());
  }, [files, localImageFiles]);

  // Reconciled Dataset mapping Memos
  const reconciledResults = React.useMemo(() => {
    const fileMap = new Map<string, { name: string; url?: string }>();
    
    // Server files
    files.forEach((file) => {
      fileMap.set(file.name, {
        name: file.name,
        url: `/api/visible-raw/${file.name}`,
      });
    });

    // Overwrite with local URLs
    localImageFiles.forEach((file) => {
      fileMap.set(file.name, {
        name: file.name,
        url: file.url,
      });
    });

    const stagedFilesMapped = Array.from(fileMap.values());
    const reconciled = reconcileDataset(predictionEntries, groundTruthEntries, stagedFilesMapped);

    if (selectedDir && selectedDir !== "Default Workspace" && selectedDir !== "Default (visible)" && selectedDir !== "default") {
      const allowedBases = new Set(stagedFilesMapped.map(sf => getTrialBase(sf.name)));
      return reconciled.filter(item => allowedBases.has(getTrialBase(item.filename)));
    }

    return reconciled;
  }, [predictionEntries, groundTruthEntries, files, localImageFiles, selectedDir]);

  const filteredCsvOutputs = React.useMemo(() => {
    if (!selectedDir || selectedDir === "Default Workspace" || selectedDir === "Default (visible)" || selectedDir === "default") {
      // Only return CSVs in the root folder, i.e. no slashes in filename
      return csvOutputs.filter(csv => !csv.filename.includes("/") && !csv.filename.includes("\\"));
    } else {
      // Return CSVs inside the selected subdirectory folder
      return csvOutputs.filter(csv => csv.filename.startsWith(selectedDir + "/") || csv.filename.startsWith(selectedDir + "\\"));
    }
  }, [csvOutputs, selectedDir]);

  const selectedResult = React.useMemo(() => {
    if (!selectedResultKey) return null;
    return reconciledResults.find((r) => r.key === selectedResultKey) || null;
  }, [reconciledResults, selectedResultKey]);

  const refreshTrainFiles = async (dirOverride?: string) => {
    try {
      const activeDir = dirOverride !== undefined ? dirOverride : selectedTrainDir;
      const url = `/api/files?dir=${encodeURIComponent(activeDir)}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setTrainFiles(data);
      }
    } catch (err) {
      console.error("Failed to fetch train files", err);
    }
  };

  const refreshInferFiles = async (dirOverride?: string) => {
    try {
      const activeDir = dirOverride !== undefined ? dirOverride : selectedInferDir;
      const url = `/api/files?dir=${encodeURIComponent(activeDir)}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setInferFiles(data);
      }
    } catch (err) {
      console.error("Failed to fetch infer files", err);
    }
  };

  const refreshDirs = async () => {
    try {
      const res = await fetch("/api/visible-directories");
      if (res.ok) {
        const data = await res.json();
        setAvailableDirs(data);
      }
    } catch (err) {
      console.error("Failed to fetch custom directories", err);
    }
  };

  // Selected config in grouped list
  const [selectedYoloVersion, setSelectedYoloVersion] = useState("8");
  const [selectedYoloSize, setSelectedYoloSize] = useState("m");

  // Terminal log options
  const [terminalFeed, setTerminalFeed] = useState<"training" | "inference">("training");
  const [workspaceFilter, setWorkspaceFilter] = useState<"mat" | "image" | "json">("mat");
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [inferenceLogs, setInferenceLogs] = useState<string[]>([]);
  const logs = terminalFeed === "training" ? trainingLogs : inferenceLogs;

  const logTerminalRef = useRef<HTMLDivElement>(null);

  // Active timing ticks for training & inference run duration tracking
  const [secondsTick, setSecondsTick] = useState(0);

  useEffect(() => {
    let interval: any = null;
    const isTrainRunning = systemStatus?.training.status === "running";
    const isInferRunning = systemStatus?.inference.status === "running";
    
    if (isTrainRunning || isInferRunning) {
      interval = setInterval(() => {
        setSecondsTick((prev) => prev + 1);
      }, 1000);
    } else {
      setSecondsTick(0);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [systemStatus?.training.status, systemStatus?.inference.status]);

  const getElapsedTimeString = (startTime: number | null | undefined, endTime: number | null | undefined) => {
    if (!startTime) return "";
    const end = endTime || Date.now();
    const diffMs = end - startTime;
    if (diffMs < 0) return "00:00";
    const diffSec = Math.floor(diffMs / 1000);
    const h = Math.floor(diffSec / 3600);
    const m = Math.floor((diffSec % 3600) / 60);
    const s = diffSec % 60;
    
    const pad = (num: number) => String(num).padStart(2, "0");
    if (h > 0) {
      return `${pad(h)}:${pad(m)}:${pad(s)}`;
    }
    return `${pad(m)}:${pad(s)}`;
  };

  // Actions states
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Fetch initial states
  const refreshFiles = async (dirOverride?: string) => {
    try {
      const activeDir = dirOverride !== undefined ? dirOverride : selectedDir;
      const url = `/api/files?dir=${encodeURIComponent(activeDir)}`;
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setFiles(data);
      }
    } catch (err) {
      console.error("Failed to fetch files", err);
    }
  };

  const refreshRuns = async () => {
    try {
      const res = await fetch("/api/runs");
      if (res.ok) {
        const data = await res.json();
        setRuns(data);
      }
    } catch (err) {
      console.error("Failed to fetch custom runs runs", err);
    }
  };

  const refreshCSVOutputs = async () => {
    try {
      const res = await fetch("/api/csv-outputs");
      if (res.ok) {
        const data = await res.json();
        setCsvOutputs(data);
      }
    } catch (err) {
      console.error("Failed to fetch generated CSV results", err);
    }
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
      }
    } catch (err) {
      console.error("Failed to fetch real-time module status", err);
    }
  };

  const refreshCondaEnvs = async (forceReload = false) => {
    setIsCondaLoading(true);
    try {
      const res = await fetch(`/api/conda/environments${forceReload ? "?reload=true" : ""}`);
      if (res.ok) {
        const data = await res.json();
        setCondaInstalled(data.condaInstalled);
        if (data.environments && Array.isArray(data.environments)) {
          setCondaEnvs(data.environments);
          
          // Highly requested priority auto-selection: ml-learning conda env
          if (data.environments.includes("ml-learning")) {
            setSelectedCondaEnv("ml-learning");
          } else if (data.environments.includes("base")) {
            setSelectedCondaEnv("base");
          } else if (data.environments.length > 0) {
            setSelectedCondaEnv(data.environments[0]);
          } else {
            setSelectedCondaEnv("system-default");
          }
        }
      }
    } catch (err) {
      console.error("Failed to fetch Conda environments", err);
    } finally {
      setIsCondaLoading(false);
    }
  };

  const fetchActiveLogs = async () => {
    // Legacy fallback, logs are now updated automatically via Server-Sent Events (SSE)
    try {
      const res = await fetch(`/api/logs/${terminalFeed}`);
      if (res.ok) {
        const data = await res.json();
        if (terminalFeed === "training") {
          setTrainingLogs(data.logs || []);
        } else {
          setInferenceLogs(data.logs || []);
        }
      }
    } catch (err) {
      console.error("Failed to fetch legacy logs for fallback", err);
    }
  };

  // Setup Server-Sent Events (SSE) listener
  useEffect(() => {
    let es: EventSource | null = null;
    let reconnectTimeout: any = null;

    const connectSSE = () => {
      console.log("Establishing Server-Sent Events (SSE) connection...");
      es = new EventSource("/api/events");

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "initial") {
            setSystemStatus(data.status);
            setTrainingLogs(data.logs.training || []);
            setInferenceLogs(data.logs.inference || []);
          } else if (data.type === "status") {
            setSystemStatus(data.status);
          } else if (data.type === "log") {
            if (data.feed === "training") {
              setTrainingLogs((prev) => {
                const updated = [...prev, data.log];
                return updated.slice(-2000);
              });
            } else if (data.feed === "inference") {
              setInferenceLogs((prev) => {
                const updated = [...prev, data.log];
                return updated.slice(-2000);
              });
            }
          } else if (data.type === "clear") {
            if (data.feed === "training") {
              setTrainingLogs([]);
            } else if (data.feed === "inference") {
              setInferenceLogs([]);
            }
          } else if (data.type === "runs_updated") {
            refreshRuns();
            refreshCSVOutputs();
            refreshFiles();
          }
        } catch (err) {
          console.error("Encountered error parsing SSE message:", err);
        }
      };

      es.onerror = (err) => {
        console.error("SSE Connection broken. Reconnecting in 3s...", err);
        if (es) {
          es.close();
        }
        reconnectTimeout = setTimeout(connectSSE, 3000);
      };
    };

    connectSSE();

    return () => {
      if (es) {
        es.close();
      }
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
      }
    };
  }, []);

  // Initial loads
  useEffect(() => {
    refreshFiles();
    refreshTrainFiles();
    refreshInferFiles();
    refreshRuns();
    refreshCSVOutputs();
    refreshCondaEnvs();
    refreshDirs();
  }, []);

  // Reload files when active directory environment changes
  useEffect(() => {
    refreshFiles();
  }, [selectedDir]);

  // Reload training directory environment changes
  useEffect(() => {
    refreshTrainFiles();
  }, [selectedTrainDir]);

  // Reload inference directory environment changes
  useEffect(() => {
    refreshInferFiles();
  }, [selectedInferDir]);

  // Sync automatic scrolling inside the pseudo terminal
  useEffect(() => {
    if (logTerminalRef.current) {
      logTerminalRef.current.scrollTop = logTerminalRef.current.scrollHeight;
    }
  }, [logs]);

  // File drag & drop helpers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      await handleFileUpload(Array.from(e.dataTransfer.files));
    }
  };

  const fileInputRef = useRef<HTMLInputElement>(null);
  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      await handleFileUpload(Array.from(e.target.files));
    }
  };

  const handleFileUpload = async (filesToUpload: File[]) => {
    setUploadMessage(`Uploading ${filesToUpload.length} file(s)...`);

    // Instantly register any images to local image state for zero-latency previews!
    const imageFiles = filesToUpload.filter(file => 
      file.type.startsWith("image/") || 
      file.name.toLowerCase().endsWith(".png") || 
      file.name.toLowerCase().endsWith(".jpg") || 
      file.name.toLowerCase().endsWith(".jpeg")
    );
    if (imageFiles.length > 0) {
      setLocalImageFiles(prev => {
        const next = [...prev];
        imageFiles.forEach(file => {
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
          const res = await fetch("/api/upload", {
            method: "POST",
            body: singleFormData,
          });
          if (res.ok) {
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
        setUploadMessage(`Staged all ${succeededCount} file(s) inside visible directory!`);
      } else {
        setUploadMessage(`Staged ${succeededCount} file(s). ${failedCount} file(s) failed.`);
      }
      refreshFiles();
      setTimeout(() => setUploadMessage(null), 4000);
    } catch (err: any) {
      setUploadMessage(`Error: ${err.message}`);
    }
  };

  // File scanner copying from custom local path
  const handleScanLocal = async () => {
    if (!localSourcePath.trim()) {
      setErrorMessage("Please supply a valid absolute or relative path to a local directory.");
      return;
    }
    setIsScanning(true);
    setLocalScanResult(null);
    setErrorMessage(null);

    try {
      const res = await fetch("/api/copy-local", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sourcePath: localSourcePath }),
      });

      const data = await res.json();
      if (res.ok) {
        setLocalScanResult(data);
        if (data.copiedCount > 0) {
          setSuccessMessage(`Found and copied ${data.copiedCount} new training assets to './visible/${data.folderName}' directory.`);
          await refreshDirs();
          if (data.folderName) {
            setSelectedDir(data.folderName);
            setSelectedTrainDir(data.folderName);
            setSelectedInferDir(data.folderName);
            refreshFiles(data.folderName);
            refreshTrainFiles(data.folderName);
            refreshInferFiles(data.folderName);
          } else {
            refreshFiles();
            refreshTrainFiles();
            refreshInferFiles();
          }
        } else {
          setErrorMessage(data.message || "No valid files found in this path directory.");
        }
      } else {
        setErrorMessage(data.error || "Directory scan and staging failed.");
      }
    } catch (err: any) {
      setErrorMessage(err.message);
    } finally {
      setIsScanning(false);
    }
  };

  // Handle deleting individual file
  const handleDeleteFile = async (filename: string) => {
    if (!window.confirm(`Permanently remove ${filename} from local storage?`)) return;
    try {
      const res = await fetch(`/api/files/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      if (res.ok) {
        setSuccessMessage(`Deleted file '${filename}'`);
        refreshFiles();
      } else {
        const err = await res.json();
        setErrorMessage(err.error || "Could not complete delete.");
      }
    } catch (err: any) {
      setErrorMessage(err.message);
    }
  };

  // Training trigger controls
  const handleStartTraining = async () => {
    setErrorMessage(null);
    setSuccessMessage(null);
    try {
      const envToSend = selectedCondaEnv === "custom" ? customCondaEnv : selectedCondaEnv;
      const res = await fetch("/api/train/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...trainConfig, condaEnv: envToSend, selectedDir: selectedTrainDir }),
      });

      if (res.ok) {
        setSuccessMessage(`Launched asynchronous YOLO training process in Conda Env [${envToSend}]. Check details on Logging Terminal.`);
        setTerminalFeed("training");
        setActiveTab("logs");
      } else {
        const err = await res.json();
        setErrorMessage(err.error || "Training startup failed.");
      }
    } catch (err: any) {
      setErrorMessage(err.message);
    }
  };

  const handleStopTraining = async () => {
    if (!window.confirm("Abort current training process? Any progress will be discarded.")) return;
    try {
      const res = await fetch("/api/train/stop", { method: "POST" });
      if (res.ok) {
        setSuccessMessage("Training aborted successfully.");
      } else {
        const err = await res.json();
        setErrorMessage(err.error || "Could not cancel processes.");
      }
    } catch (err: any) {
      setErrorMessage(err.message);
    }
  };

  // Inference trigger controls
  const handleStartInference = async () => {
    setErrorMessage(null);
    setSuccessMessage(null);
    try {
      const envToSend = selectedCondaEnv === "custom" ? customCondaEnv : selectedCondaEnv;
      const res = await fetch("/api/infer/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...inferConfig, condaEnv: envToSend, selectedDir: selectedInferDir }),
      });

      if (res.ok) {
        setSuccessMessage(`Inference process triggered on staged MAT files in Conda Env [${envToSend}]. Results exporting to CSV.`);
        setTerminalFeed("inference");
        setActiveTab("logs");
      } else {
        const err = await res.json();
        setErrorMessage(err.error || "Inference execution failed.");
      }
    } catch (err: any) {
      setErrorMessage(err.message);
    }
  };

  const handleStopInference = async () => {
    try {
      const res = await fetch("/api/infer/stop", { method: "POST" });
      if (res.ok) {
        setSuccessMessage("Inference process halted.");
      } else {
        const err = await res.json();
        setErrorMessage(err.error || "Could not halt inference.");
      }
    } catch (err: any) {
      setErrorMessage(err.message);
    }
  };

  // Delete run configuration references
  const handleDeleteRunRef = async (id: string) => {
    if (!window.confirm("Remove this configuration record from catalog?")) return;
    try {
      const res = await fetch(`/api/runs/${id}`, { method: "DELETE" });
      if (res.ok) {
        setSuccessMessage("Ref reference cleared.");
        refreshRuns();
      }
    } catch (err: any) {
      setErrorMessage(err.message);
    }
  };

  const handleLoadCsvIntoInspector = async (associatedCSVUrl: string) => {
    setErrorMessage(null);
    setSuccessMessage(null);
    try {
      const res = await fetch(associatedCSVUrl);
      if (!res.ok) {
        throw new Error("Unable to fetch CSV file output from server storage.");
      }
      const csvText = await res.text();
      setPredictionsCsvText(csvText);
      const parsedPreds = parsePredictions(csvText);
      setPredictionEntries(parsedPreds);
      
      const parts = associatedCSVUrl.split("/");
      const filename = parts[parts.length - 1];
      setPredictionsFilename(filename || "results.csv");
      
      // Auto-extract subdirectory if loaded via download URL
      const urlPrefix = "/api/download/csv/";
      if (associatedCSVUrl.startsWith(urlPrefix)) {
        const fileSubPath = associatedCSVUrl.substring(urlPrefix.length);
        if (fileSubPath.includes("/")) {
          const folder = fileSubPath.split("/")[0];
          if (availableDirs.includes(folder)) {
            setSelectedDir(folder);
          }
        }
      }

      // Auto-select the first item in the reconciled list if available
      if (parsedPreds.length > 0) {
        const firstPredName = parsedPreds[0].filename;
        const base = getTrialBase(firstPredName);
        const angle = normalizeAngle(parsedPreds[0].angle ?? extractAngle(firstPredName));
        setSelectedResultKey(`${base}::${angle}`);
      }
      
      setSuccessMessage("Successfully parsed and loaded YOLO predictions CSV into VisionTrack Inspector!");
      setActiveTab("indigo-600"); // Let's set activeTab to "inspector"
      setActiveTab("inspector");
    } catch (err: any) {
      setErrorMessage(`Failed to load CSV: ${err.message}`);
    }
  };

  const handleDeleteCsvOutput = async (filename: string) => {
    if (!window.confirm(`Are you sure you want to permanently delete the results file "${filename}"?`)) return;
    try {
      const res = await fetch(`/api/csv-outputs/${filename}`, { method: "DELETE" });
      if (res.ok) {
        setSuccessMessage("Permanently deleted CSV result file.");
        refreshCSVOutputs();
        refreshRuns();
      } else {
        const err = await res.json();
        setErrorMessage(err.error || "Could not delete CSV file.");
      }
    } catch (err: any) {
      setErrorMessage(err.message);
    }
  };

  // Filters for displaying lists
  const matFiles = files.filter((f) => f.type === "mat");
  const jsonLabels = files.filter((f) => f.type === "json");
  const imageFiles = files.filter((f) => f.type === "image");

  const trainMatFiles = trainFiles.filter((f) => f.type === "mat");
  const trainJsonLabels = trainFiles.filter((f) => f.type === "json");
  const inferMatFiles = inferFiles.filter((f) => f.type === "mat");

  return (
    <div id="dashboard_panel" className={`min-h-screen bg-slate-50 text-slate-900 font-sans selection:bg-indigo-500 selection:text-white ${activeTab === "inspector" ? "" : "pb-12"}`}>
      
      {/* HEADER BAR */}
      {activeTab !== "inspector" && (
        <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
          <div className="max-w-7xl mx-auto px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              <div className="bg-indigo-600 p-2 rounded-lg text-white shadow-sm shadow-indigo-100">
                <Brain className="w-6 h-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight text-slate-800 flex items-center gap-2">
                  Pseudopupil YOLO Backend Console
                  <span className="text-xs font-mono font-medium px-2 py-0.5 bg-slate-100 text-slate-500 rounded-full">v1.0.0</span>
                </h1>
                <p className="text-xs text-slate-500 font-mono">
                  Port 3000 Pipeline Agent • Offline Training & Centroid Refinement Diagnostics
                </p>
              </div>
            </div>

            {/* Navigation Controls & Status Connected */}
            <div className="flex flex-wrap gap-4 items-center text-sm font-medium">
              <div className="flex bg-slate-100 border border-slate-200 rounded-lg p-1 gap-1">
                <button
                  id="tab_dashboard"
                  onClick={() => setActiveTab("dashboard")}
                  className={`px-4 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center gap-2 cursor-pointer ${
                    activeTab === "dashboard"
                      ? "bg-white text-indigo-700 shadow-sm border border-slate-200/50"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <Settings className="w-3.5 h-3.5" />
                  Machine Station
                </button>
                <button
                  id="tab_inspector"
                  onClick={() => setActiveTab("inspector")}
                  className={`px-4 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center gap-2 cursor-pointer ${
                    activeTab === "inspector"
                      ? "bg-white text-indigo-700 shadow-sm border border-slate-200/50"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  <Layers className="w-3.5 h-3.5" />
                  VisionTrack Inspector
                  {reconciledResults.length > 0 && (
                    <span className="bg-indigo-100 text-indigo-700 text-[9px] px-1.5 py-0.2 rounded-full font-mono font-extrabold ml-0.5">
                      {reconciledResults.length}
                    </span>
                  )}
                </button>
                <button
                  id="tab_logs"
                  onClick={() => {
                    setActiveTab("logs");
                    fetchActiveLogs();
                  }}
                  className={`px-4 py-1.5 text-xs font-semibold rounded-md transition-all flex items-center gap-2 cursor-pointer ${
                    activeTab === "logs"
                      ? "bg-white text-indigo-700 shadow-sm border border-slate-200/50"
                      : "text-slate-600 hover:text-slate-900 relative"
                  }`}
                >
                  <Terminal className="w-3.5 h-3.5" />
                  Live Process Logs
                  {(systemStatus?.training.status === "running" || systemStatus?.inference.status === "running") && (
                    <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full animate-ping" />
                  )}
                </button>
              </div>
              <div className="hidden md:block h-8 w-px bg-slate-200"></div>
              <div className="flex items-center gap-2 bg-slate-100 px-3 py-1.5 rounded-full text-xs text-slate-700">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span>Server Connected</span>
              </div>
            </div>
          </div>
        </header>
      )}

      {/* SYSTEM BROADCAST AREA */}
      {activeTab !== "inspector" && (
        <div className="max-w-7xl mx-auto px-4 mt-6 animate-fade-in">
          {errorMessage && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-800 rounded-lg text-xs flex items-center gap-3 shadow-2xs">
              <AlertTriangle className="w-4 h-4 text-red-600 flex-shrink-0" />
              <div className="flex-1 font-medium">{errorMessage}</div>
              <button onClick={() => setErrorMessage(null)} className="text-red-400 hover:text-red-700 font-bold px-1.5 transition">✕</button>
            </div>
          )}

          {successMessage && (
            <div className="mb-4 p-3 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-lg text-xs flex items-center gap-3 shadow-2xs">
              <CheckCircle className="w-4 h-4 text-emerald-600 flex-shrink-0" />
              <div className="flex-1 font-medium">{successMessage}</div>
              <button onClick={() => setSuccessMessage(null)} className="text-emerald-400 hover:text-emerald-700 font-bold px-1.5 transition">✕</button>
            </div>
          )}
        </div>
      )}

      <main className={activeTab === "inspector" ? "w-full min-h-screen bg-[#f8fafc]" : "max-w-7xl mx-auto px-4 pb-12"}>
        {activeTab === "dashboard" ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            
            {/* COLUMN LEFT: FILE EXPLORER & STAGING PANEL (4 cols) */}
            <div className="lg:col-span-4 flex flex-col gap-6">
              
              {/* STAGING CONTROLLER (FileUpload + Directory Scan) */}
              <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
                <h2 className="text-sm font-semibold text-slate-800 mb-3 flex items-center gap-2 tracking-tight">
                  <Database className="w-4 h-4 text-indigo-600" />
                  Staging Configuration
                </h2>
                <p className="text-xs text-slate-550 mb-4 tracking-wide leading-relaxed">
                  Stage raw <code className="text-slate-700 bg-slate-50 border border-slate-200/60 px-1.5 py-0.5 rounded text-[11px] font-mono">.mat</code> imagery inputs and annotations to the active 
                  <code className="text-slate-700 bg-slate-50 border border-slate-200/60 px-1.5 py-0.5 rounded text-[11px] font-mono ml-1">visible</code> workspace directory.
                </p>

                {/* Local Folder Scan & Copy Mirror (Matching "copy file from a directory in local storage to this visible folder") */}
                <div className="border border-slate-200 bg-slate-50 p-3.5 rounded-lg mb-4">
                  <label className="block text-xs font-mono font-semibold text-slate-700 mb-1.5 flex items-center gap-1">
                    <FolderOpen className="w-3.5 h-3.5 text-indigo-600" />
                    Mirror Local Server Directory:
                  </label>
                  <p className="text-[10px] text-slate-500 mb-2 font-mono">
                    Avoid web browser upload entirely by syncing directly from your disk folder candidate path.
                  </p>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      className="flex-1 bg-white text-slate-800 border border-slate-300 rounded px-2.5 py-1.5 text-xs font-mono placeholder:text-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-2xs"
                      placeholder="/home/user/my_mat_dataset"
                      value={localSourcePath}
                      onChange={(e) => setLocalSourcePath(e.target.value)}
                    />
                    <button
                      onClick={handleScanLocal}
                      disabled={isScanning}
                      className="bg-indigo-50 hover:bg-indigo-100 disabled:opacity-50 text-indigo-700 border border-indigo-200 rounded px-3 py-1 text-xs transition-all flex items-center gap-1.5 font-mono font-semibold cursor-pointer"
                    >
                      {isScanning ? (
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      ) : (
                        <ArrowRight className="w-3.5 h-3.5" />
                      )}
                      Sync
                    </button>
                  </div>
                </div>

                {/* Upload Zone */}
                <div
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-lg p-5 text-center cursor-pointer transition-all ${
                    dragActive
                      ? "border-indigo-500 bg-indigo-50/50 text-indigo-700"
                      : "border-slate-200 hover:border-slate-300 hover:bg-slate-50 text-slate-500 bg-white"
                  }`}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileSelect}
                    multiple
                    onClick={(e) => e.stopPropagation()}
                    className="hidden"
                    accept=".mat,.json,.png,.jpg,.jpeg"
                  />
                  <Download className="w-6 h-6 mx-auto mb-2 text-slate-400 animate-pulse" />
                  <p className="text-xs font-semibold text-slate-700">
                    Drag & Drop or <span className="text-indigo-600 underline decoration-indigo-600/30">Browse</span>
                  </p>
                  <p className="text-[10px] text-slate-450 mt-1 font-mono">
                    Accepts MAT, JSON, PNG, JPG, JPEG files
                  </p>
                </div>

                {uploadMessage && (
                  <div className="mt-2 text-center text-[10px] font-mono p-1 rounded bg-indigo-50/50 border border-indigo-100 text-indigo-700">
                    {uploadMessage}
                  </div>
                )}
              </div>

              {/* STAGED FILES EXPLORER */}
              <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col">
                <div className="flex items-center justify-between mb-3 pb-1 border-b border-slate-100">
                  <h3 className="text-sm font-semibold text-slate-800 flex items-center gap-2 tracking-tight">
                    <FileCode className="w-4 h-4 text-indigo-600" />
                    Visible Area Workspace
                  </h3>
                  <button 
                    onClick={refreshFiles}
                    className="p-1 hover:bg-slate-100 inline-flex items-center text-slate-500 hover:text-slate-800 rounded transition cursor-pointer"
                  >
                    <RefreshCw className="w-3 h-3" />
                  </button>
                </div>

                <div className="grid grid-cols-3 gap-2 text-center mb-4 pb-2 border-b border-dashed border-slate-150">
                  <div className="bg-slate-50/50 rounded-lg p-1.5 border border-slate-200/50">
                    <div className="text-[10px] uppercase font-bold text-slate-400">MAT</div>
                    <div className="text-xs font-bold text-slate-700 font-mono">{matFiles.length}</div>
                  </div>
                  <div className="bg-slate-50/50 rounded-lg p-1.5 border border-slate-200/50">
                    <div className="text-[10px] uppercase font-bold text-slate-400">Images</div>
                    <div className="text-xs font-bold text-slate-700 font-mono">{imageFiles.length}</div>
                  </div>
                  <div className="bg-slate-50/50 rounded-lg p-1.5 border border-slate-200/50">
                    <div className="text-[10px] uppercase font-bold text-slate-400">Labels</div>
                    <div className="text-xs font-bold text-slate-700 font-mono">{jsonLabels.length}</div>
                  </div>
                </div>

                <div className="mb-3.5">
                  <label htmlFor="workspace_dir_select" className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1.5 flex items-center gap-1">
                    <FolderOpen className="w-3.5 h-3.5 text-indigo-500" />
                    Select Dataset Directory
                  </label>
                  <select
                    id="workspace_dir_select"
                    value={selectedDir}
                    onChange={(e) => setSelectedDir(e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 text-slate-700 text-xs rounded-lg px-2.5 py-2 font-semibold focus:outline-hidden focus:border-indigo-500 transition shadow-2xs font-mono"
                  >
                    {availableDirs.map((name) => (
                      <option key={name} value={name}>
                        {name === "Default Workspace" ? "📁 Default Workspace (visible)" : `📁 ${name}`}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="mb-4">
                  <label htmlFor="workspace_category_select" className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1.5">
                    Select Filter Category
                  </label>
                  <select
                    id="workspace_category_select"
                    value={workspaceFilter}
                    onChange={(e) => setWorkspaceFilter(e.target.value as "mat" | "image" | "json")}
                    className="w-full bg-slate-50 border border-slate-200 text-slate-700 text-xs rounded-lg px-2.5 py-2 font-semibold focus:outline-hidden focus:border-indigo-500 transition shadow-2xs"
                  >
                    <option value="mat">📄 MAT Files ({matFiles.length})</option>
                    <option value="image">🖼️ Images ({imageFiles.length})</option>
                    <option value="json">🔬 Labels ({jsonLabels.length})</option>
                  </select>
                </div>

                {files.length === 0 ? (
                  <div className="text-center py-8 border border-slate-150 rounded bg-slate-50/50">
                    <Database className="w-6 h-6 text-slate-300 mx-auto mb-2" />
                    <p className="text-xs text-slate-450 font-mono">No files staged yet.</p>
                  </div>
                ) : files.filter((f) => f.type === workspaceFilter).length === 0 ? (
                  <div className="text-center py-6 border border-slate-150 rounded bg-slate-50/20">
                    <p className="text-xs text-slate-400 font-mono">No files in this category.</p>
                  </div>
                ) : (
                  <div className="max-h-[380px] overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                    {files.filter((f) => f.type === workspaceFilter).map((file) => (
                      <div
                        key={file.name}
                        className="flex items-center justify-between p-2.5 bg-slate-50/50 border border-slate-200 rounded-lg hover:border-slate-300 hover:bg-white text-xs transition shadow-2xs"
                      >
                        <div className="flex items-center gap-2.5 min-w-0">
                          <code className={`px-1.5 py-0.5 text-[9px] rounded font-bold uppercase ${
                            file.type === "mat"
                              ? "bg-indigo-50 border border-indigo-100 text-indigo-600"
                              : file.type === "json"
                              ? "bg-amber-50 border border-amber-100 text-amber-700"
                              : "bg-emerald-50 border border-emerald-100 text-emerald-700"
                          }`}>
                            {file.type === "json" ? "label" : file.type}
                          </code>
                          <div className="min-w-0">
                            <div className="font-mono text-slate-700 truncate pr-2 font-semibold" title={file.name}>
                              {file.name}
                            </div>
                            <div className="text-[10px] text-slate-400 font-mono mt-0.5">
                              {(file.size / 1024 / 1024).toFixed(2)} MB • {new Date(file.mtime).toLocaleDateString()}
                            </div>
                          </div>
                        </div>

                        <button
                          onClick={() => handleDeleteFile(file.name)}
                          className="p-1 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded transition cursor-pointer"
                        >
                          <Trash className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* COLUMN RIGHT: CONTROLLERS & EXECUTORS Gird (8 cols) */}
            <div className="lg:col-span-8 flex flex-col gap-6">
              
              {/* CONDA ENVIRONMENT SELECT CARD */}
              <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm relative overflow-hidden">
                {isCondaLoading && (
                  <div className="absolute inset-0 bg-slate-100/70 backdrop-blur-[1.5px] flex flex-col items-center justify-center z-15 transition-all duration-300">
                    <div className="flex flex-col items-center gap-2 bg-white/90 px-4 py-2.5 rounded-lg shadow-sm border border-slate-200/50 animate-pulse">
                      <RefreshCw className="w-5 h-5 animate-spin text-indigo-600" />
                      <span className="text-[11px] font-semibold text-slate-700 font-mono">Loading Conda Environments...</span>
                    </div>
                  </div>
                )}
                <div className="flex items-center justify-between mb-3 border-b border-slate-100 pb-2">
                  <h2 className="text-sm font-semibold text-slate-800 flex items-center gap-2 tracking-tight">
                    <Settings className="w-4 h-4 text-indigo-600" />
                    Python Execution Environment (Conda)
                  </h2>
                  <div className="flex items-center gap-1.5">
                    {isCondaLoading ? (
                      <span className="text-[10px] text-slate-450 font-mono flex items-center gap-1 animate-pulse">
                        <RefreshCw className="w-3 h-3 animate-spin text-indigo-500" /> Scanning...
                      </span>
                    ) : (
                      <button
                        onClick={() => refreshCondaEnvs(true)}
                        className="text-[10px] text-indigo-600 hover:underline font-semibold font-mono flex items-center gap-1 cursor-pointer"
                        title="Query conda environments again"
                      >
                        <RefreshCw className="w-2.5 h-2.5" /> Reload List
                      </button>
                    )}
                  </div>
                </div>

                <div className="space-y-3">
                  <p className="text-xs text-slate-500 leading-relaxed font-sans">
                    Select the active environment to execute your YOLO runs. The server dynamically activates the selected environment when running training or inference tasks.
                  </p>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-end">
                    <div>
                      <label className="block text-[10px] font-mono font-bold text-slate-400 mb-1 uppercase tracking-wider">
                        Active Conda Target:
                      </label>
                      <select
                        value={selectedCondaEnv}
                        onChange={(e) => setSelectedCondaEnv(e.target.value)}
                        className="w-full bg-slate-50 border border-slate-350 rounded px-2.5 py-1.5 text-xs font-mono text-slate-800 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-2xs cursor-pointer"
                      >
                        <option value="system-default">System Default Python PATH</option>
                        {condaEnvs.map((env) => (
                          <option key={env} value={env}>
                            Conda: {env} {env === "ml-learning" ? "★" : ""}
                          </option>
                        ))}
                        <option value="custom">Custom (Type Environment Name)</option>
                      </select>
                    </div>

                    <div>
                      {selectedCondaEnv === "custom" ? (
                        <div>
                          <label className="block text-[10px] font-mono font-bold text-slate-400 mb-1 uppercase tracking-wider">
                            Enter Env Name:
                          </label>
                          <input
                            type="text"
                            value={customCondaEnv}
                            onChange={(e) => setCustomCondaEnv(e.target.value)}
                            placeholder="e.g. ml-learning"
                            className="w-full bg-white text-slate-850 border border-slate-350 rounded px-2.5 py-1.5 text-xs font-mono placeholder:text-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-2xs"
                          />
                        </div>
                      ) : (
                        <div className="bg-slate-50 border border-slate-200/60 rounded p-2 text-[10.5px] font-mono text-slate-500">
                          <span className="font-semibold text-slate-650">Active Context:</span>{" "}
                          <span className="text-indigo-600 font-extrabold text-[11px]">
                            {selectedCondaEnv === "system-default" ? "global-python" : selectedCondaEnv}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {!condaInstalled && (
                    <div className="mt-1 flex items-start gap-1.5 bg-amber-50 border border-amber-200 p-2 rounded text-[10.5px] text-amber-800 font-mono leading-tight">
                      <span className="font-bold">⚠️ Discovery Mode:</span>
                      <span>
                        Conda CLI wasn't auto-detected on host system. Custom environment paths can be manually typed above if initialized under alias parameters.
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* TRAINING & INFERENCE BENTO ROW Grid */}
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                       {/* 1. TRAINING CONTROLLER CARD */}
                <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-3 border-b border-slate-100 pb-2">
                      <h2 className="text-sm font-semibold text-slate-800 flex items-center gap-2 tracking-tight">
                        <Layers className="w-4 h-4 text-indigo-600" />
                        YOLOv8/11/26 Trainer
                      </h2>
                      {systemStatus?.training.status === "running" ? (
                        <div className="flex items-center gap-2">
                          {systemStatus.training.startTime && (
                            <span className="text-[10px] font-mono font-bold text-emerald-700 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-100 flex items-center gap-1">
                              <Clock className="w-3 h-3 text-emerald-600" />
                              {getElapsedTimeString(systemStatus.training.startTime, systemStatus.training.endTime)}
                            </span>
                          )}
                          <span className="flex items-center gap-1.5 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-full px-2.5 py-0.5 text-[9px] font-mono font-bold animate-pulse">
                            ● Training
                          </span>
                        </div>
                      ) : systemStatus?.training.startTime ? (
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] font-mono font-semibold text-slate-550 bg-slate-105 px-1.5 py-0.5 rounded border border-slate-200/60 flex items-center gap-1">
                            <Clock className="w-3 h-3 text-slate-400" />
                            {getElapsedTimeString(systemStatus.training.startTime, systemStatus.training.endTime)}
                          </span>
                          <span className={`px-2 py-0.5 text-[9px] font-mono font-bold rounded-full uppercase border ${
                            systemStatus.training.status === 'completed'
                              ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                              : systemStatus.training.status === 'aborted'
                              ? 'bg-amber-50 text-amber-805 border-amber-200'
                              : 'bg-rose-50 text-rose-800 border-rose-200'
                          }`}>
                            {systemStatus.training.status}
                          </span>
                        </div>
                      ) : (
                        <span className="text-xs font-mono font-bold text-slate-400 uppercase">Idle</span>
                      )}
                    </div>

                    <div className="flex flex-col gap-3 py-1">
                      
                      {/* Dataset Directory selection for Training */}
                      <div>
                        <label className="block text-[10px] font-mono font-bold text-slate-500 mb-1 uppercase tracking-wider">Training Dataset directory:</label>
                        <select
                          value={selectedTrainDir}
                          onChange={(e) => setSelectedTrainDir(e.target.value)}
                          disabled={systemStatus?.training.status === "running"}
                          className="w-full bg-slate-50 border border-slate-350 rounded px-2.5 py-1.5 text-xs font-mono text-slate-805 font-bold focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-2xs cursor-pointer"
                        >
                          {availableDirs.map((name) => (
                            <option key={name} value={name}>
                              {name === "Default Workspace" ? "📁 Default Workspace (visible)" : `📁 ${name}`}
                            </option>
                          ))}
                        </select>
                      </div>

                      {/* YOLO Version selection */}
                      <div>
                        <label className="block text-[10px] font-mono font-bold text-slate-500 mb-1 uppercase tracking-wider">YOLO Engine Version:</label>
                        <div className="grid grid-cols-3 gap-1">
                          {["8", "11", "26"].map((v) => (
                            <button
                              key={v}
                              onClick={() => setTrainConfig({ ...trainConfig, version: v })}
                              disabled={systemStatus?.training.status === "running"}
                              className={`py-1.5 text-xs font-mono font-semibold border rounded transition cursor-pointer ${
                                trainConfig.version === v
                                  ? "bg-indigo-600 border-indigo-600 text-white shadow-sm shadow-indigo-100"
                                  : "border-slate-200 hover:border-slate-300 bg-slate-50 text-slate-600 hover:text-slate-850 disabled:opacity-50"
                              }`}
                            >
                              YOLOv{v}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Model size selector */}
                      <div>
                        <label className="block text-[10px] font-mono font-bold text-slate-500 mb-1 uppercase tracking-wider">Model Dimensions Size:</label>
                        <div className="grid grid-cols-4 gap-1">
                          {["n", "s", "m", "l"].map((sz) => (
                            <button
                              key={sz}
                              onClick={() => setTrainConfig({ ...trainConfig, size: sz })}
                              disabled={systemStatus?.training.status === "running"}
                              className={`py-1.5 text-xs font-mono font-semibold border rounded capitalize transition cursor-pointer ${
                                trainConfig.size === sz
                                  ? "bg-indigo-600 border-indigo-600 text-white shadow-sm shadow-indigo-100"
                                  : "border-slate-200 hover:border-slate-300 bg-slate-50 text-slate-600 hover:text-slate-850 disabled:opacity-50"
                              }`}
                            >
                              {sz === "n" ? "Nano" : sz === "s" ? "Small" : sz === "m" ? "Med" : "Large"}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Selectable annotations JSON file */}
                      <div>
                        <label className="block text-[10px] font-mono font-bold text-slate-500 mb-1 uppercase tracking-wider">Label Annotation Source (JSON):</label>
                        {trainJsonLabels.length === 0 ? (
                          <div className="p-2.5 border border-amber-200 bg-amber-50 text-amber-800 rounded text-[10px] font-mono flex items-center gap-1.5 shadow-2xs leading-relaxed">
                            <Info className="w-3.5 h-3.5 text-amber-600 flex-shrink-0" />
                            No label JSON in visible/labels/ yet. Defaults to root placeholder.
                          </div>
                        ) : (
                          <select
                            value={trainConfig.labelFile}
                            disabled={systemStatus?.training.status === "running"}
                            onChange={(e) => setTrainConfig({ ...trainConfig, labelFile: e.target.value })}
                            className="w-full bg-slate-50 border border-slate-350 rounded px-2.5 py-1.5 text-xs font-mono text-slate-800 placeholder:text-slate-400 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-2xs disabled:opacity-50"
                          >
                            <option value="">-- Autoselect lowest matching JSON --</option>
                            {trainJsonLabels.map((lbl) => (
                              <option key={lbl.name} value={lbl.name}>
                                {lbl.name}
                              </option>
                            ))}
                          </select>
                        )}
                      </div>

                      {/* Active Workspace path indicator */}
                      <div className="mt-1 text-[10px] text-indigo-700 px-2 py-1.5 bg-indigo-50/40 border border-indigo-100/40 rounded flex justify-between font-mono">
                        <span className="text-slate-500 font-sans font-medium">Staging target directory:</span>
                        <span className="font-bold truncate max-w-[200px]">
                          {selectedTrainDir === "Default Workspace" ? "visible/" : `visible/${selectedTrainDir}/`}
                        </span>
                      </div>

                      {/* Custom input numbers: Epochs & Batches */}
                      <div className="grid grid-cols-2 gap-3 mt-1">
                        <div>
                          <label className="block text-[10px] font-mono font-bold text-slate-500 mb-1 uppercase tracking-wider">Epoch Limit:</label>
                          <input
                            type="number"
                            disabled={systemStatus?.training.status === "running"}
                            value={trainConfig.epochs}
                            min={1}
                            max={1000}
                            onChange={(e) => setTrainConfig({ ...trainConfig, epochs: parseInt(e.target.value) || 10 })}
                            className="w-full bg-slate-50 border border-slate-350 rounded px-2.5 py-1.5 text-xs font-mono text-slate-800 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-2xs disabled:opacity-50"
                          />
                        </div>
                        <div>
                          <label className="block text-[10px] font-mono font-bold text-slate-500 mb-1 uppercase tracking-wider">Batch Size:</label>
                          <input
                            type="number"
                            disabled={systemStatus?.training.status === "running"}
                            value={trainConfig.batch}
                            min={1}
                            max={128}
                            onChange={(e) => setTrainConfig({ ...trainConfig, batch: parseInt(e.target.value) || 16 })}
                            className="w-full bg-slate-50 border border-slate-350 rounded px-2.5 py-1.5 text-xs font-mono text-slate-800 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-2xs disabled:opacity-50"
                          />
                        </div>
                      </div>

                      {/* Hardware accelerator selection */}
                      <div className="mt-1">
                        <label className="block text-[10px] font-mono font-bold text-slate-500 mb-1 uppercase tracking-wider">Hardware Device:</label>
                        <div className="flex gap-2">
                          {["cpu", "cuda"].map((dev) => (
                            <label key={dev} className="flex-1 flex items-center justify-center gap-1.5 border border-slate-200 bg-slate-50 p-2 rounded text-xs font-mono cursor-pointer hover:border-slate-300 hover:bg-slate-100/50 transition duration-150">
                              <input
                                type="radio"
                                name="deviceSelect"
                                disabled={systemStatus?.training.status === "running"}
                                checked={trainConfig.device === dev}
                                onChange={() => setTrainConfig({ ...trainConfig, device: dev })}
                                className="accent-indigo-600 w-3.5 h-3.5"
                              />
                              <span className="uppercase font-semibold text-slate-700">{dev}</span>
                            </label>
                          ))}
                        </div>
                      </div>

                    </div>
                  </div>

                  {/* Asynchronous Executor Start and Stop controls for trainer */}
                  <div className="mt-5 border-t border-slate-100 pt-4">
                    {systemStatus?.training.status === "running" ? (
                      <div className="space-y-4">
                        {/* Interactive live metrics & progress percentage bar */}
                        <div>
                          <div className="flex justify-between text-xs mb-1 font-mono font-bold">
                            <span className="text-slate-500">Epoch {systemStatus.training.currentEpoch}/{systemStatus.training.totalEpochs}</span>
                            <div className="flex items-center gap-1.5 text-[11px] text-slate-505">
                              <span className="font-normal font-sans">Time Elapsed: {getElapsedTimeString(systemStatus.training.startTime, systemStatus.training.endTime)}</span>
                              <span>|</span>
                              <span className="text-indigo-600 font-bold">{systemStatus.training.percentage}%</span>
                            </div>
                          </div>
                          <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden border border-slate-200">
                            <div
                              className="bg-indigo-600 h-full rounded-full transition-all duration-300"
                              style={{ width: `${systemStatus.training.percentage}%` }}
                            />
                          </div>
                          <p className="text-[10px] text-slate-400 mt-1 font-mono truncate" title={systemStatus.training.metrics}>
                            Metrics: {systemStatus.training.metrics}
                          </p>
                        </div>

                        <button
                          onClick={handleStopTraining}
                          className="w-full py-2 bg-rose-50 hover:bg-rose-100 border border-rose-200 text-rose-700 rounded-lg text-xs font-mono font-bold flex items-center justify-center gap-2 transition duration-300 cursor-pointer shadow-2xs"
                        >
                          <Square className="w-3.5 h-3.5 fill-rose-700" />
                          Abort Train Process (Early Stop)
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={handleStartTraining}
                        disabled={trainMatFiles.length === 0}
                        className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-100 disabled:text-slate-400 text-white rounded-lg font-semibold text-xs font-mono flex items-center justify-center gap-2 transition duration-300 shadow-xs cursor-pointer"
                      >
                        <Play className="w-3.5 h-3.5 fill-white" />
                        Execute Training Run
                      </button>
                    )}
                  </div>
                </div>

                {/* 2. INFERENCE RUN CONTROL CARD */}
                <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-3 border-b border-slate-100 pb-2">
                      <h2 className="text-sm font-semibold text-slate-800 flex items-center gap-2 tracking-tight">
                        <Cpu className="w-4 h-4 text-indigo-600" />
                        Classical Refinement Infer
                      </h2>
                      {systemStatus?.inference.status === "running" ? (
                        <div className="flex items-center gap-2">
                          {systemStatus.inference.startTime && (
                            <span className="text-[10px] font-mono font-bold text-indigo-700 bg-indigo-50 px-1.5 py-0.5 rounded border border-indigo-100 flex items-center gap-1">
                              <Clock className="w-3 h-3 text-indigo-500" />
                              {getElapsedTimeString(systemStatus.inference.startTime, systemStatus.inference.endTime)}
                            </span>
                          )}
                          <span className="flex items-center gap-1.5 bg-indigo-50 border border-indigo-200 text-indigo-750 rounded-full px-2.5 py-0.5 text-[9px] font-mono font-bold animate-pulse">
                            ● Inferencing
                          </span>
                        </div>
                      ) : systemStatus?.inference.startTime ? (
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] font-mono font-semibold text-slate-550 bg-slate-105 px-1.5 py-0.5 rounded border border-slate-200/60 flex items-center gap-1">
                            <Clock className="w-3 h-3 text-slate-400" />
                            {getElapsedTimeString(systemStatus.inference.startTime, systemStatus.inference.endTime)}
                          </span>
                          <span className={`px-2 py-0.5 text-[9px] font-mono font-bold rounded-full uppercase border ${
                            systemStatus.inference.status === 'completed'
                              ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
                              : systemStatus.inference.status === 'aborted'
                              ? 'bg-amber-50 text-amber-805 border-amber-200'
                              : 'bg-rose-50 text-rose-800 border-rose-200'
                          }`}>
                            {systemStatus.inference.status}
                          </span>
                        </div>
                      ) : (
                        <span className="text-xs font-mono font-bold text-slate-400 uppercase">Idle</span>
                      )}
                    </div>

                     <div className="flex flex-col gap-3 py-1">
                      <p className="text-xs text-slate-500 tracking-wide leading-relaxed">
                        Evaluates files in <code className="text-slate-700 bg-slate-50 border border-slate-200 px-1 py-0.5 rounded text-[11px] font-mono">visible/</code> using the specific, trained oriented bounding box 
                        <code className="text-slate-700 bg-slate-50 border border-slate-200 px-1.5 py-0.5 rounded text-[11px] font-mono ml-1">best.pt</code> model. Outputs final geometric centroid measurements to CSV format of results.
                      </p>

                      {/* Dataset Directory selection for Inference */}
                      <div>
                        <label className="block text-[10px] font-mono font-bold text-slate-500 mb-1 uppercase tracking-wider">Inference Dataset directory:</label>
                        <select
                          value={selectedInferDir}
                          onChange={(e) => setSelectedInferDir(e.target.value)}
                          disabled={systemStatus?.inference.status === "running"}
                          className="w-full bg-slate-50 border border-slate-350 rounded px-2.5 py-1.5 text-xs font-mono text-indigo-800 font-bold focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-2xs cursor-pointer"
                        >
                          {availableDirs.map((name) => (
                            <option key={name} value={name}>
                              {name === "Default Workspace" ? "📁 Default Workspace (visible)" : `📁 ${name}`}
                            </option>
                          ))}
                        </select>
                      </div>

                      <div className="space-y-3 mt-1">
                        
                        {/* Match infer script target configuration models */}
                        <div>
                          <label className="block text-[10px] font-mono font-bold text-slate-500 mb-1.5 uppercase tracking-wider">Target Model Config:</label>
                          <div className="grid grid-cols-2 gap-2">
                            <div>
                              <label className="block text-[9px] font-mono font-bold text-slate-450 mb-0.5">YOLO Version</label>
                              <select
                                value={inferConfig.version}
                                disabled={systemStatus?.inference.status === "running"}
                                onChange={(e) => setInferConfig({ ...inferConfig, version: e.target.value })}
                                className="w-full bg-slate-50 border border-slate-350 rounded px-2.5 py-1.5 text-xs font-mono text-slate-800 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-2xs disabled:opacity-50"
                              >
                                <option value="8">YOLOv8</option>
                                <option value="11">YOLOv11</option>
                                <option value="26">YOLOv26</option>
                              </select>
                            </div>
                            <div>
                              <label className="block text-[9px] font-mono font-bold text-slate-450 mb-0.5">Model Size</label>
                              <select
                                value={inferConfig.size}
                                disabled={systemStatus?.inference.status === "running"}
                                onChange={(e) => setInferConfig({ ...inferConfig, size: e.target.value })}
                                className="w-full bg-slate-50 border border-slate-350 rounded px-2.5 py-1.5 text-xs font-mono text-slate-800 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 shadow-2xs disabled:opacity-50"
                              >
                                <option value="n">Nano</option>
                                <option value="s">Small</option>
                                <option value="m">Medium</option>
                                <option value="l">Large</option>
                              </select>
                            </div>
                          </div>
                        </div>

                        <div>
                          <label className="block text-[10px] font-mono font-bold text-slate-500 mb-1 uppercase tracking-wider">Processor Device:</label>
                          <div className="flex gap-2">
                            {["cpu", "cuda"].map((dev) => (
                              <label key={dev} className="flex-1 flex items-center justify-center gap-1.5 border border-slate-200 bg-slate-50 p-2 rounded text-xs font-mono cursor-pointer hover:border-slate-300 hover:bg-slate-100/50 transition duration-150">
                                <input
                                  type="radio"
                                  name="inferDeviceSelect"
                                  disabled={systemStatus?.inference.status === "running"}
                                  checked={inferConfig.device === dev}
                                  onChange={() => setInferConfig({ ...inferConfig, device: dev })}
                                  className="accent-indigo-600 w-3.5 h-3.5"
                                />
                                <span className="uppercase font-semibold text-slate-700">{dev}</span>
                              </label>
                            ))}
                          </div>
                        </div>

                        {/* File status summary inside staging dir */}
                        <div className="border border-slate-200 p-3 bg-slate-50 rounded font-mono text-[10px] space-y-1 mt-2">
                          <div className="flex justify-between">
                            <span className="text-slate-555">Staged Inputs Found:</span>
                            <span className={inferMatFiles.length > 0 ? "text-slate-800 font-bold" : "text-amber-600 font-bold"}>{inferMatFiles.length} file(s)</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-550 font-sans">Active Target Directory:</span>
                            <span className="text-indigo-700 font-bold truncate max-w-[190px]">
                              {selectedInferDir === "Default Workspace" ? "visible/" : `visible/${selectedInferDir}/`}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-slate-550 font-sans">Predicted weights directory:</span>
                            <span className="text-slate-650 truncate max-w-[190px]">runs/pseudopupil_yolo{inferConfig.version}{inferConfig.size}_obb/...</span>
                          </div>
                        </div>

                      </div>
                    </div>
                    
                    {/* 📁 REAL-TIME DETECTED INFERENCE RESULTS CSV WINDOW */}
                    <div className="mt-3.5 pt-3 border-t border-slate-105">
                      <div className="flex items-center justify-between mb-2">
                        <span className="block text-[10px] font-mono font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1.5 text-indigo-650 font-extrabold animate-pulse">
                          <FileSpreadsheet className="w-3.5 h-3.5 text-indigo-505" />
                          Inference Results Registry Windows
                        </span>
                        <span className="text-[9px] font-mono font-bold text-indigo-650 bg-indigo-50 px-1.5 py-0.5 rounded-sm border border-indigo-100">
                          {filteredCsvOutputs.length} run(s)
                        </span>
                      </div>
                      
                      {filteredCsvOutputs.length === 0 ? (
                        <div className="border border-dashed border-slate-205 rounded-lg p-3 bg-slate-50/50 text-center">
                          <span className="block text-[10.5px] text-slate-400 font-mono italic leading-relaxed">
                            No result CSV sheets generated yet. After running inference, results will automatically reflect here.
                          </span>
                        </div>
                      ) : (
                        <div className="border border-slate-200/80 rounded-lg bg-slate-50/10 max-h-[160px] overflow-y-auto divide-y divide-slate-100 shadow-2xs">
                          {[...filteredCsvOutputs].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()).map((csv) => {
                            const formattedSize = (csv.size / 1024).toFixed(1) + " KB";
                            const isNew = (Date.now() - new Date(csv.timestamp).getTime()) < 30000;
                            return (
                              <div key={csv.filename} className={`p-2 flex items-center justify-between gap-2.5 transition hover:bg-slate-50/80 ${isNew ? 'bg-indigo-50/30' : ''}`}>
                                <div className="min-w-0 flex-1">
                                  <div className="flex items-center gap-1.5">
                                    <span className={`block text-[10.5px] font-mono font-bold text-slate-700 truncate ${isNew ? 'text-indigo-700 animate-pulse' : ''}`} title={csv.filename}>
                                      {csv.filename.includes("/") ? csv.filename.split("/").pop() : csv.filename.includes("\\") ? csv.filename.split("\\").pop() : csv.filename}
                                    </span>
                                    {isNew && (
                                      <span className="inline-block flex-shrink-0 animate-pulse bg-emerald-600 text-white font-extrabold text-[8px] px-1 py-0.2 rounded-xs uppercase">
                                        Recently Generated
                                      </span>
                                    )}
                                  </div>
                                  <div className="flex items-center gap-1.5 mt-0.5 text-[9px] text-slate-400 font-mono">
                                    <span>{formattedSize}</span>
                                    <span>•</span>
                                    <span>{new Date(csv.timestamp).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' })}</span>
                                  </div>
                                </div>
                                <div className="flex items-center gap-1 flex-shrink-0">
                                  <button
                                    onClick={() => handleLoadCsvIntoInspector(`/api/download/csv/${csv.filename}`)}
                                    className="px-1.5 py-0.5 bg-white hover:bg-indigo-50 text-slate-650 hover:text-indigo-700 border border-slate-200 hover:border-indigo-200 rounded text-[9.5px] font-mono font-semibold transition cursor-pointer"
                                    title="Load predictions CSV directly in Visual Track Inspector"
                                  >
                                    Inspect
                                  </button>
                                  <a
                                    href={`/api/download/csv/${csv.filename}`}
                                    download
                                    className="p-1 text-slate-400 hover:text-slate-700 rounded transition cursor-pointer"
                                    title="Download CSV locally"
                                  >
                                    <Download className="w-3 h-3" />
                                  </a>
                                  <button
                                    onClick={() => handleDeleteCsvOutput(csv.filename)}
                                    className="p-1 text-slate-400 hover:text-rose-600 rounded transition cursor-pointer"
                                    title="Delete CSV permanently"
                                  >
                                    <Trash className="w-3 h-3" />
                                  </button>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Asynchronous Executor control for inference */}
                  <div className="mt-5 border-t border-slate-100 pt-4">
                    {systemStatus?.inference.status === "running" ? (
                      <div className="space-y-4">
                        <div>
                          <div className="flex justify-between text-xs mb-1 font-mono font-bold">
                            <span className="text-slate-500">Scanning File {systemStatus.inference.currentFile}/{systemStatus.inference.totalFiles}</span>
                            <div className="flex items-center gap-1.5 text-[11px] text-slate-505">
                              <span className="font-normal font-sans">Time Elapsed: {getElapsedTimeString(systemStatus.inference.startTime, systemStatus.inference.endTime)}</span>
                              <span>|</span>
                              <span className="text-indigo-600 font-bold">{systemStatus.inference.percentage}%</span>
                            </div>
                          </div>
                          <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden border border-slate-200">
                            <div
                              className="bg-indigo-600 h-full rounded-full transition-all duration-300"
                              style={{ width: `${systemStatus.inference.percentage}%` }}
                            />
                          </div>
                          <p className="text-[10px] text-slate-450 mt-1 font-mono truncate" title={systemStatus.inference.currentFileName}>
                            Active File: {systemStatus.inference.currentFileName || "Awaiting file sync..."}
                          </p>
                        </div>

                        <button
                          onClick={handleStopInference}
                          className="w-full py-2 bg-slate-100 hover:bg-slate-200 border border-slate-250 text-slate-700 hover:text-slate-900 rounded-lg text-xs font-mono font-bold flex items-center justify-center gap-2 transition duration-350 cursor-pointer"
                        >
                          <Square className="w-3.5 h-3.5 fill-slate-700" />
                          Halt Inference Run
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={handleStartInference}
                        disabled={inferMatFiles.length === 0}
                        className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-100 disabled:text-slate-400 text-white rounded-lg font-semibold text-xs font-mono flex items-center justify-center gap-2 transition duration-300 shadow-xs cursor-pointer"
                      >
                        <Play className="w-3.5 h-3.5 fill-white" />
                        Execute Inference Run
                      </button>
                    )}
                  </div>
                </div>

              </div>

              {/* CORE GROUPED OUTPUT HISTORY EXPLORER */}
              <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
                <div className="flex items-center justify-between mb-4 border-b border-slate-100 pb-2.5">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-indigo-600" />
                    <h3 className="text-sm font-semibold text-slate-800 tracking-tight">
                      Grouped Training Models & Associated Predictions
                    </h3>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => {
                        refreshRuns();
                        refreshCSVOutputs();
                      }}
                      className="px-2.5 py-1 text-xs text-slate-600 hover:text-slate-905 border border-slate-200 bg-slate-50 hover:bg-slate-100/50 rounded-md font-mono transition flex items-center gap-1.5 cursor-pointer"
                    >
                      <RefreshCw className="w-3 h-3" />
                      Reload Catalog
                    </button>
                  </div>
                </div>

                <p className="text-xs text-slate-500 font-sans tracking-wide mb-4 leading-relaxed">
                  As training completes, we record is stored inside local catalog logs. To run inference, the weights for YOLO matching version and size will be executed.
                </p>

                {runs.length === 0 ? (
                  <div className="text-center py-10 border border-slate-150 rounded bg-slate-50/50">
                    <Database className="w-8 h-8 text-slate-350 mx-auto mb-2.5" />
                    <p className="text-xs font-semibold text-slate-500 font-mono">No active trained models recorded in local catalog yet.</p>
                    <p className="text-[11px] text-slate-400 font-mono mt-1">Start a training session to register an oriented bbox weights model.</p>
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse text-left text-xs font-mono">
                      <thead>
                        <tr className="border-b border-slate-200 text-slate-450 text-[10px] uppercase font-bold tracking-wider">
                          <th className="py-2.5 px-3">Config Signature</th>
                          <th className="py-2.5 px-3">Trained Weights File</th>
                          <th className="py-2.5 px-3">Refinement Output (CSV)</th>
                          <th className="py-2.5 px-3 text-right">Settings</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {runs.map((run) => (
                          <tr key={run.id} className="hover:bg-slate-50/50 transition-all">
                            <td className="py-3 px-3">
                              <div className="flex items-center gap-2">
                                <span className="w-2 h-2 bg-emerald-500 rounded-full" />
                                <span className="text-slate-800 font-bold uppercase">{run.modelConfig}</span>
                              </div>
                              <span className="text-[10px] text-slate-400 block mt-0.5">
                                Synced {new Date(run.timestamp).toLocaleString()}
                              </span>
                            </td>
                            <td className="py-3 px-3">
                              {run.weightsFile ? (
                                <div className="space-y-1">
                                  <div className="text-indigo-600 font-bold truncate max-w-[200px]" title={run.weightsFile}>
                                    best.pt
                                  </div>
                                  <a
                                    href={`/api/download/weights/${run.modelConfig}`}
                                    className="inline-flex items-center gap-1.5 text-[10px] text-slate-500 hover:text-indigo-600 underline decoration-slate-300"
                                  >
                                    <Download className="w-3 h-3" />
                                    Download Weights
                                  </a>
                                </div>
                              ) : (
                                <span className="text-slate-400 italic font-normal">None generated</span>
                              )}
                            </td>
                            <td className="py-3 px-3">
                              {run.associatedCSV ? (
                                <div className="space-y-1">
                                  <div className="text-slate-800 font-bold flex items-center gap-1.5">
                                    <FileSpreadsheet className="w-3.5 h-3.5 text-indigo-500" />
                                    results.csv
                                  </div>
                                  <div className="flex items-center gap-2.5">
                                    <a
                                      href={run.associatedCSV}
                                      download
                                      className="inline-flex items-center gap-1.5 text-[10px] text-slate-500 hover:text-indigo-600 underline decoration-slate-300"
                                    >
                                      <Download className="w-3 h-3" />
                                      Download CSV
                                    </a>
                                    <span className="text-slate-250">|</span>
                                    <button
                                      onClick={() => handleLoadCsvIntoInspector(run.associatedCSV!)}
                                      className="inline-flex items-center gap-1.5 text-[10px] text-indigo-600 hover:text-indigo-800 font-bold hover:underline cursor-pointer"
                                    >
                                      <Layers className="w-3 h-3 text-indigo-505" />
                                      Inspect Live
                                    </button>
                                  </div>
                                </div>
                              ) : (
                                <span className="text-slate-400 italic font-normal">No runs executed</span>
                              )}
                            </td>
                            <td className="py-3 px-3 text-right">
                              <button
                                onClick={() => handleDeleteRunRef(run.id)}
                                className="p-1 hover:bg-rose-50 text-slate-400 hover:text-rose-600 rounded transition duration-200 cursor-pointer"
                                title="Remove configuration reference"
                              >
                                <Trash className="w-3.5 h-3.5" />
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

            </div>

          </div>
        ) : activeTab === "inspector" ? (
          <VisionTrackInspectorPanel
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            predictionEntries={predictionEntries}
            setPredictionEntries={setPredictionEntries}
            predictionsFilename={predictionsFilename}
            setPredictionsFilename={setPredictionsFilename}
            groundTruthEntries={groundTruthEntries}
            setGroundTruthEntries={setGroundTruthEntries}
            trainingLogEntries={trainingLogEntries}
            setTrainingLogEntries={setTrainingLogEntries}
            reconciledResults={reconciledResults}
            selectedResultKey={selectedResultKey}
            setSelectedResultKey={setSelectedResultKey}
            selectedResult={selectedResult}
            modifiedPredictions={modifiedPredictions}
            setModifiedPredictions={setModifiedPredictions}
            modifiedCoarse={modifiedCoarse}
            setModifiedCoarse={setModifiedCoarse}
            invertY={invertY}
            setInvertY={setInvertY}
            yOffset={yOffset}
            setYOffset={setYOffset}
            showCoarse={showCoarse}
            setShowCoarse={setShowCoarse}
            showGroundTruth={showGroundTruth}
            setShowGroundTruth={setShowGroundTruth}
            showPredictions={showPredictions}
            setShowPredictions={setShowPredictions}
            showGrid={showGrid}
            setShowGrid={setShowGrid}
            csvOutputs={filteredCsvOutputs}
            files={mergedFiles}
            refreshFiles={refreshFiles}
            localImageFiles={localImageFiles}
            setLocalImageFiles={setLocalImageFiles}
            availableDirs={availableDirs}
            selectedDir={selectedDir}
            setSelectedDir={setSelectedDir}
          />
        ) : (
          
          /* LIVE DETAILED PROCESS LOGS AND PSEUDO TERMINAL SCREEN */
          <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col gap-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-100 pb-3 gap-3">
              <div className="flex items-center gap-2">
                <Terminal className="w-5 h-5 text-indigo-600" />
                <h3 className="text-sm font-semibold text-slate-800 tracking-tight">
                  Console Logging Interface
                </h3>
              </div>

              <div className="flex bg-slate-100 border border-slate-200 p-0.5 rounded-lg gap-0.5">
                <button
                  onClick={() => setTerminalFeed("training")}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-md transition duration-150 cursor-pointer ${
                    terminalFeed === "training"
                      ? "bg-white text-indigo-700 shadow-2xs border border-slate-200/50"
                      : "text-slate-500 hover:text-slate-800"
                  }`}
                >
                  Training Proc
                </button>
                <button
                  onClick={() => setTerminalFeed("inference")}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-md transition duration-150 cursor-pointer ${
                    terminalFeed === "inference"
                      ? "bg-white text-indigo-700 shadow-2xs border border-slate-200/50"
                      : "text-slate-500 hover:text-slate-800"
                  }`}
                >
                  Inference Proc
                </button>
              </div>
            </div>

            <p className="text-xs text-slate-500 tracking-wide">
              Live standard output logging streams spawned directly from python processes executing under background worker limits.
            </p>

            {/* Pseudo high-performance monochrome terminal block */}
            <div
              ref={logTerminalRef}
              className="bg-slate-950 font-mono text-[11px] text-slate-300 p-4 rounded-xl border border-slate-900 h-[480px] overflow-y-auto space-y-1.5 custom-scrollbar shadow-inner"
            >
              {logs.length === 0 ? (
                <div className="text-center py-20 text-slate-500 font-mono">
                  &gt;_ No actively buffered logs generated yet. trigger training or inference to display streams.
                </div>
              ) : (
                logs.map((log, index) => {
                  let logColor = "text-slate-300";
                  if (log.startsWith("[STDERR]") || log.includes("Error:") || log.toLowerCase().includes("failed")) {
                    logColor = "text-rose-450";
                  } else if (log.startsWith("EPOCH_PROGRESS") || log.toLowerCase().includes("success") || log.startsWith("FILE_PROGRESS")) {
                    logColor = "text-emerald-450 font-semibold";
                  } else if (log.includes("🚀") || log.includes("🔥") || log.includes("🎬")) {
                    logColor = "text-amber-450";
                  }
                  
                  return (
                    <div key={index} className="leading-5 whitespace-pre-wrap select-text selection:bg-indigo-600 selection:text-white">
                      <span className="text-slate-600 mr-2 select-none">[{index + 1}]</span>
                      <span className={logColor}>{log}</span>
                    </div>
                  );
                })
              )}
            </div>

            <div className="flex justify-between items-center bg-slate-50 p-3 rounded-lg border border-slate-200 text-xs font-mono text-slate-500">
              <div className="flex items-center gap-2">
                <Clock className="w-3.5 h-3.5 text-slate-400" />
                <span>Active terminal target: <span className="text-slate-850 font-bold uppercase">{terminalFeed}</span></span>
              </div>
              <button
                onClick={() => {
                  if (terminalFeed === "training") {
                    setTrainingLogs([]);
                  } else {
                    setInferenceLogs([]);
                  }
                }}
                className="px-2.5 py-1 hover:bg-slate-100 text-slate-605 hover:text-slate-850 rounded border border-slate-200 transition cursor-pointer font-semibold shadow-2xs bg-white"
              >
                Clear Terminal Buffer
              </button>
            </div>
          </div>

        )}
      </main>

    </div>
  );
}
