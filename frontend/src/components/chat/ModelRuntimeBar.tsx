import { Cpu, Loader2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { LLMSettings } from "@/lib/api";

type OpenAICodexModel =
  | "openai-codex/gpt-5.6-sol"
  | "openai-codex/gpt-5.6-luna"
  | "openai-codex/gpt-6-sol"
  | "openai-codex/gpt-6-luna";

interface Props {
  settings: LLMSettings | null;
  runtimeProvider?: string;
  runtimeModel?: string;
  runtimeReasoningEffort?: string;
  switching?: boolean;
  switchDisabled?: boolean;
  onProviderSwitch?: (provider: "deepseek" | "openai-codex") => void;
  onChatGptModelChange?: (model: OpenAICodexModel) => void;
  onReasoningEffortChange?: (effort: "" | "none" | "low" | "medium" | "high" | "max") => void;
}

export function ModelRuntimeBar({
  settings,
  runtimeProvider,
  runtimeModel,
  runtimeReasoningEffort,
  switching = false,
  switchDisabled = false,
  onProviderSwitch,
  onChatGptModelChange,
  onReasoningEffortChange,
}: Props) {
  const { t } = useTranslation();
  if (!settings) return null;

  const providerId = runtimeProvider || settings.provider;
  const provider = settings.providers.find((item) => item.name === providerId);
  const providerLabel = provider?.label || providerId || t("agent.unknownProvider");
  const model = runtimeModel || settings.model_name || t("agent.unknownModel");
  const effortLabels: Record<string, string> = {
    none: t("settings.reasoningEffortNone"),
    low: t("settings.reasoningEffortLow"),
    medium: t("settings.reasoningEffortMedium"),
    high: t("settings.reasoningEffortHigh"),
    max: t("settings.reasoningEffortMax"),
  };
  const reasoningEffort = runtimeReasoningEffort !== undefined
    ? runtimeReasoningEffort
    : settings.reasoning_effort;
  const effortLabel = effortLabels[reasoningEffort] || t("settings.providerDefault");
  const configuredProvider = settings.provider === "openai-codex" ? "openai-codex" : "deepseek";
  const knownChatGptModel = (
    [
      "openai-codex/gpt-5.6-sol",
      "openai-codex/gpt-5.6-luna",
      "openai-codex/gpt-6-sol",
      "openai-codex/gpt-6-luna",
    ] as const
  ).includes(settings.model_name as OpenAICodexModel);
  const chatGptModel = settings.model_name;
  const selectableEffort = ["", "none", "low", "medium", "high", "max"].includes(settings.reasoning_effort)
    ? settings.reasoning_effort
    : "";
  const canSwitch = Boolean(
    onProviderSwitch
    && settings.providers.some((item) => item.name === "deepseek")
    && settings.providers.some((item) => item.name === "openai-codex"),
  );

  return (
    <div className="shrink-0 border-b border-border/70 bg-background/95 px-6 py-2 backdrop-blur-sm">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-2 text-xs sm:flex-nowrap">
        <span className="relative flex h-2 w-2 shrink-0" aria-hidden="true">
          <span className="absolute inline-flex h-full w-full rounded-full bg-success/30" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-success" />
        </span>
        <span className="shrink-0 font-medium text-foreground">{providerLabel}</span>
        <span className="text-muted-foreground/60">·</span>
        <span className="truncate font-mono text-[11px] text-muted-foreground" title={model}>{model}</span>
        {canSwitch && (
          <div
            className="ml-auto inline-flex shrink-0 items-center rounded-full border border-border/70 bg-muted/35 p-0.5"
            role="group"
            aria-label={t("agent.modelSwitchLabel")}
          >
            {(["deepseek", "openai-codex"] as const).map((candidate) => {
              const selected = configuredProvider === candidate;
              const label = candidate === "deepseek" ? "DeepSeek" : "ChatGPT";
              return (
                <button
                  key={candidate}
                  type="button"
                  aria-pressed={selected}
                  disabled={switchDisabled || switching}
                  title={switchDisabled ? t("agent.modelSwitchBusyHint") : t("agent.modelSwitchTo", { provider: label })}
                  onClick={() => {
                    if (!selected) onProviderSwitch?.(candidate);
                  }}
                  className={[
                    "inline-flex h-5 items-center gap-1 rounded-full px-2 text-[10px] font-medium transition-colors",
                    selected
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground",
                    "disabled:cursor-not-allowed disabled:opacity-50",
                  ].join(" ")}
                >
                  {switching && !selected && <Loader2 className="h-2.5 w-2.5 animate-spin" aria-hidden="true" />}
                  {label}
                </button>
              );
            })}
          </div>
        )}
        {configuredProvider === "openai-codex" && onChatGptModelChange && (
          <select
            aria-label={`${t("settings.modelName")} ChatGPT`}
            title={t("settings.modelName")}
            value={chatGptModel}
            disabled={switchDisabled || switching}
            onChange={(event) => onChatGptModelChange(event.target.value as OpenAICodexModel)}
            className="h-6 shrink-0 rounded-md border border-border/70 bg-background px-1.5 text-[10px] text-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            {!knownChatGptModel && <option value={chatGptModel}>{chatGptModel}</option>}
            <option value="openai-codex/gpt-5.6-sol">5.6 Sol</option>
            <option value="openai-codex/gpt-5.6-luna">5.6 Luna</option>
            <option value="openai-codex/gpt-6-sol">6 Sol</option>
            <option value="openai-codex/gpt-6-luna">6 Luna</option>
          </select>
        )}
        {configuredProvider === "openai-codex" && onReasoningEffortChange && (
          <select
            aria-label={t("settings.reasoningEffort")}
            title={t("settings.reasoningEffort")}
            value={selectableEffort}
            disabled={switchDisabled || switching}
            onChange={(event) => onReasoningEffortChange(
              event.target.value as "" | "none" | "low" | "medium" | "high" | "max",
            )}
            className="h-6 shrink-0 rounded-md border border-border/70 bg-background px-1.5 text-[10px] text-foreground disabled:cursor-not-allowed disabled:opacity-50"
          >
            <option value="">{t("settings.providerDefault")}</option>
            <option value="none">{effortLabels.none}</option>
            <option value="low">{effortLabels.low}</option>
            <option value="medium">{effortLabels.medium}</option>
            <option value="high">{effortLabels.high}</option>
            <option value="max">{effortLabels.max}</option>
          </select>
        )}
        <span className={[
          "inline-flex shrink-0 items-center gap-1.5 rounded-full border border-border/70 bg-muted/35 px-2 py-0.5 text-[10px] text-muted-foreground",
          canSwitch ? "" : "ml-auto",
        ].join(" ")}>
          <Cpu className="h-3 w-3" aria-hidden="true" />
          {t("agent.reasoningStrength")}: {effortLabel}
        </span>
      </div>
    </div>
  );
}
