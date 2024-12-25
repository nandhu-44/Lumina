import React from "react";
import { Loader2 } from "lucide-react";
import { motion } from "framer-motion";

const GenerationProgress = ({ progress }) => {
    if (!progress) return null;

    return (
        <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
            <div className="space-y-3">
                <div className="flex justify-between items-center">
                    <div className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin text-purple-400" />
                        <span className="text-sm text-gray-300">
                            Generating image...
                        </span>
                    </div>
                    <span className="text-sm text-purple-400 font-medium">
                        {Math.round(progress.progress)}%
                    </span>
                </div>

                <div className="h-2 bg-gray-700/50 rounded-full overflow-hidden">
                    <motion.div
                        className="h-full bg-gradient-to-r from-purple-500 to-pink-500"
                        initial={{ width: 0 }}
                        animate={{ width: `${progress.progress}%` }}
                        transition={{ duration: 0.3 }}
                    />
                </div>

                <div className="grid grid-cols-2 gap-4 text-xs">
                    <div className="space-y-1">
                        <div className="text-gray-500">Progress</div>
                        <div className="text-gray-300">
                            Step {progress.step} of {progress.total_steps}
                        </div>
                    </div>
                    <div className="space-y-1 text-right">
                        <div className="text-gray-500">Time</div>
                        <div className="text-gray-300">
                            {progress.estimated_time > 0
                                ? `~${
                                    Math.round(progress.estimated_time)
                                }s remaining`
                                : `${
                                    Math.round(progress.elapsed_time)
                                }s elapsed`}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default GenerationProgress;
