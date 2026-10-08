export type ReasoningEffort = "" | "none" | "low" | "medium" | "high" | "xhigh" | "max";

export function reasoningEffortsForModel(model: string): ReasoningEffort[] {
  const efforts: ReasoningEffort[] = ["", "none", "low", "medium", "high", "xhigh", "max"];
  return model.split("/").slice(-1)[0] === "gpt-6.1-sol"
    ? efforts.filter((effort) => effort !== "none")
    : efforts;
}

export function normalizeReasoningEffort(model: string, effort: string): string {
  return model.split("/").slice(-1)[0] === "gpt-6.1-sol" && ["none", "minimal"].includes(effort)
    ? "medium"
    : effort;
}

export const reasoningEffortTranslationKeys = {
  "": "settings.providerDefault",
  none: "settings.reasoningEffortNone",
  low: "settings.reasoningEffortLow",
  medium: "settings.reasoningEffortMedium",
  high: "settings.reasoningEffortHigh",
  xhigh: "settings.reasoningEffortXhigh",
  max: "settings.reasoningEffortMax",
} as const;
