"use client";

import { useCallback, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import {
  Download,
  Image as ImageIcon,
  Info,
  Loader2,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useWebSocket } from "@/hooks/useWebSocket";
import { Slider } from "@/components/ui/slider";
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { nanoid } from "nanoid";
import Image from "next/image";
import GenerationProgress from "@/components/ui/generation-progress";

const defaultConfig = {
  num_inference_steps: 30,
  guidance_scale: 7.5,
  height: 512,
  width: 512,
};

const validRanges = {
  num_inference_steps: { min: 20, max: 50, step: 1 },
  guidance_scale: { min: 1, max: 10, step: 0.5 },
  height: { min: 384, max: 768, step: 128 },
  width: { min: 384, max: 768, step: 128 },
};

const controlDescriptions = {
  num_inference_steps:
    "Controls the number of denoising steps (20-50). Higher values may improve quality but increase generation time.",
  guidance_scale:
    "Controls how closely the image follows the prompt (1-10). Values around 7.5 usually work best.",
  height:
    "The height of the generated image (384-768px). Must be a multiple of 128.",
  width:
    "The width of the generated image (384-768px). Must be a multiple of 128.",
};

const controlWarnings = {
  num_inference_steps:
    "Higher values will significantly increase generation time.",
  guidance_scale:
    "Values above 8.0 may produce overly saturated or unrealistic results.",
  height:
    "Larger dimensions will exponentially increase memory usage and generation time.",
  width:
    "Larger dimensions will exponentially increase memory usage and generation time.",
};

export default function Home() {
  const [config, setConfig] = useState(defaultConfig);
  const [prompt, setPrompt] = useState("");
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [enhancedPrompt, setEnhancedPrompt] = useState("");
  const [status, setStatus] = useState("");
  const [progress, setProgress] = useState(null);

  const handleWebSocketMessage = useCallback((event) => {
    try {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "prompt":
          setEnhancedPrompt(data.data);
          break;
        case "status":
          setStatus(data.data);
          break;
        case "error":
          setError(data.data);
          setLoading(false);
          break;
        case "progress":
          setProgress(data.data);
          break;
        default:
          console.log("Unknown message type:", data.type);
      }
    } catch (error) {
      console.error("Failed to handle WebSocket message:", error);
    }
  }, []);

  useWebSocket("ws://localhost:8000/ws", handleWebSocketMessage);

  const generateImage = async () => {
    try {
      setLoading(true);
      setError(null);
      setImage(null);
      setEnhancedPrompt("");
      setStatus("");
      setProgress(null);

      const response = await fetch("http://localhost:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, ...config }),
      });

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.message);
      }

      setImage(`data:image/png;base64,${data.image}`);
    } catch (err) {
      setError(err.message);
      setProgress(null);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!image) return;

    const link = document.createElement("a");
    link.href = image;
    const id = nanoid(8);
    link.download = `enhanced-generation-${id}-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const resetConfig = () => {
    setConfig(defaultConfig);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-4 sm:p-8">
      <main className="max-w-6xl mx-auto">
        <motion.div className="text-center space-y-4 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center space-y-4"
          >
            <div className="flex items-center justify-center gap-1">
              <Image src="/lumina.png" width={64} height={64} alt="Lumina" />
              <h1 className="text-4xl sm:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-pink-600">
                Lumina
              </h1>
            </div>
            <p className="text-gray-400 text-lg">
              Transform your imagination into art with AI enhanced prompts.
            </p>
          </motion.div>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-6">
            <Card className="p-6 bg-gray-800/50 backdrop-blur border-gray-700">
              <div className="space-y-6">
                <div className="space-y-4">
                  <div className="flex gap-2">
                    <Input
                      placeholder="Describe your image..."
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      className="bg-gray-700/50 border-gray-600"
                    />
                    <Button
                      onClick={generateImage}
                      disabled={loading || !prompt}
                      className="min-w-[120px] bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white"
                    >
                      {loading ? <Loader2 className="animate-spin" /> : (
                        <>
                          <Sparkles className="mr-2 h-4 w-4 text-white" />
                          Generate
                        </>
                      )}
                    </Button>
                  </div>
                  <AnimatePresence>
                    {enhancedPrompt && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="text-sm text-gray-400 bg-gray-700/30 p-3 rounded-md"
                      >
                        <span className="font-semibold text-purple-400">
                          Enhanced:
                        </span>{" "}
                        {enhancedPrompt}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>

                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <h3 className="text-lg font-semibold">
                      Generation Settings
                    </h3>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={resetConfig}
                      className="text-gray-400 hover:text-gray-100"
                    >
                      <RotateCcw className="h-4 w-4 mr-2" />
                      Reset
                    </Button>
                  </div>
                  {Object.entries(validRanges).map(([key, range]) => (
                    <div key={key} className="space-y-2">
                      <div className="flex justify-between">
                        <div className="flex items-center gap-2">
                          <label className="text-sm text-gray-400">
                            {key.split("_").map((word) =>
                              word.charAt(0).toUpperCase() + word.slice(1)
                            ).join(" ")}
                          </label>
                          <HoverCard>
                            <HoverCardTrigger>
                              <Info className="h-4 w-4 text-gray-500 hover:text-gray-300 hover:cursor-help  " />
                            </HoverCardTrigger>
                            <HoverCardContent className="w-80 bg-gray-800 border-gray-700 text-gray-100">
                              <p className="text-sm">
                                {controlDescriptions[key]}
                              </p>
                              {controlWarnings[key] && (
                                <p className="text-xs text-amber-400 mt-1">
                                  ⚠️ {controlWarnings[key]}
                                </p>
                              )}
                              <p className="text-xs text-gray-400 mt-1">
                                Range: {range.min} - {range.max}
                                {range.step ? `, Step: ${range.step}` : ""}
                              </p>
                            </HoverCardContent>
                          </HoverCard>
                        </div>
                        <span className="text-sm text-gray-400">
                          {config[key]}
                        </span>
                      </div>
                      <Slider
                        value={[config[key]]}
                        min={range.min}
                        max={range.max}
                        step={range.step}
                        onValueChange={([value]) =>
                          setConfig((prev) => ({ ...prev, [key]: value }))}
                      />
                    </div>
                  ))}
                </div>
              </div>
            </Card>

            <AnimatePresence>
              {status && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-blue-400 text-sm bg-blue-900/20 p-3 rounded-md"
                >
                  {status}
                </motion.div>
              )}
            </AnimatePresence>
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="text-red-400 text-sm bg-red-900/20 p-3 rounded-md"
                >
                  {error}
                </motion.div>
              )}
            </AnimatePresence>
            <AnimatePresence>
              {loading && progress && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <GenerationProgress progress={progress} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <Card className="p-6 bg-gray-800/50 backdrop-blur border-gray-700">
            <AnimatePresence mode="wait">
              {image
                ? (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    className="relative w-full max-w-md mx-auto"
                  >
                    <div className="aspect-square rounded-lg overflow-hidden shadow-2xl">
                      <img
                        src={image}
                        alt="Generated image"
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <Button
                      onClick={handleDownload}
                      className="absolute bottom-4 right-4 bg-gray-900/80 hover:bg-gray-900 text-white flex items-center gap-2"
                      variant="secondary"
                    >
                      <Download className="h-4 w-4" />
                      Save
                    </Button>
                  </motion.div>
                )
                : (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="w-full max-w-md mx-auto aspect-square rounded-lg border-2 border-dashed border-gray-700 flex items-center justify-center"
                  >
                    <div className="text-gray-500 flex flex-col items-center">
                      <ImageIcon className="h-12 w-12 mb-2" />
                      <span>Your image will appear here</span>
                    </div>
                  </motion.div>
                )}
            </AnimatePresence>
          </Card>
        </div>
      </main>
    </div>
  );
}
