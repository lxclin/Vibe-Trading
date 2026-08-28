import { fireEvent, render, screen } from "@testing-library/react";
import { ModelRuntimeBar } from "../ModelRuntimeBar";
import type { LLMSettings } from "@/lib/api";

const settings: LLMSettings = {
  provider: "deepseek",
  model_name: "deepseek-v4-flash",
  base_url: "https://api.deepseek.com",
  api_key_configured: true,
  api_key_required: true,
  temperature: 0,
  timeout_seconds: 120,
  max_retries: 2,
  reasoning_effort: "high",
  sse_timeout_seconds: 90,
  env_path: "agent/.env",
  providers: [
    {
      name: "deepseek",
      label: "DeepSeek",
      base_url_env: "DEEPSEEK_BASE_URL",
      default_model: "deepseek-v4-pro",
      default_base_url: "https://api.deepseek.com/v1",
      api_key_required: true,
    },
    {
      name: "openai-codex",
      label: "OpenAI Codex (ChatGPT OAuth)",
      base_url_env: "OPENAI_CODEX_BASE_URL",
      default_model: "openai-codex/gpt-5.6-sol",
      default_base_url: "https://chatgpt.com/backend-api/codex/responses",
      api_key_required: false,
      auth_type: "oauth",
    },
  ],
};

describe("ModelRuntimeBar", () => {
  it("shows provider, provider-reported model and configured reasoning effort", () => {
    render(
      <ModelRuntimeBar
        settings={settings}
        runtimeProvider="deepseek"
        runtimeModel="deepseek-v4-flash-202607"
      />,
    );

    expect(screen.getByText("DeepSeek")).toBeInTheDocument();
    expect(screen.getByText("deepseek-v4-flash-202607")).toBeInTheDocument();
    expect(screen.getByText(/Reasoning Effort: High/)).toBeInTheDocument();
  });

  it("distinguishes explicit no-reasoning from the provider default", () => {
    render(
      <ModelRuntimeBar
        settings={{ ...settings, reasoning_effort: "none" }}
      />,
    );

    expect(screen.getByText(/Reasoning Effort: None \(explicit\)/)).toBeInTheDocument();
  });

  it("shows the persisted historical reasoning effort instead of current settings", () => {
    render(
      <ModelRuntimeBar
        settings={{ ...settings, reasoning_effort: "low" }}
        runtimeProvider="deepseek"
        runtimeModel="deepseek-v4-flash-202607"
        runtimeReasoningEffort="high"
      />,
    );

    expect(screen.getByText(/Reasoning Effort: High/)).toBeInTheDocument();
    expect(screen.queryByText(/Reasoning Effort: Low/)).not.toBeInTheDocument();
  });

  it("switches the configured provider while preserving historical runtime identity", () => {
    const onProviderSwitch = vi.fn();
    render(
      <ModelRuntimeBar
        settings={{ ...settings, provider: "openai-codex", model_name: "openai-codex/gpt-5.6-sol" }}
        runtimeProvider="deepseek"
        runtimeModel="deepseek-v4-flash"
        onProviderSwitch={onProviderSwitch}
      />,
    );

    expect(screen.getByText("DeepSeek", { selector: "span" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "ChatGPT" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: "DeepSeek" }));
    expect(onProviderSwitch).toHaveBeenCalledWith("deepseek");
  });

  it("disables both model choices while a response is running", () => {
    render(
      <ModelRuntimeBar
        settings={settings}
        switchDisabled
        onProviderSwitch={vi.fn()}
      />,
    );

    expect(screen.getByRole("button", { name: "DeepSeek" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "ChatGPT" })).toBeDisabled();
  });
});
