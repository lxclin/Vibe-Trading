import { act, cleanup, render } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router";
import { Agent } from "../Agent";
import { useAgentStore } from "@/stores/agent";

const apiMock = vi.hoisted(() => ({
  getGoal: vi.fn(), getLLMSettings: vi.fn(), getRun: vi.fn(),
  getSessionMessages: vi.fn(), sseUrl: vi.fn((sid: string) => `/sessions/${sid}/events`),
}));
vi.mock("@/lib/api", async () => ({
  ...await vi.importActual<typeof import("@/lib/api")>("@/lib/api"), api: apiMock,
}));
const sseMock = vi.hoisted(() => ({ connect: vi.fn(), disconnect: vi.fn(), onStatusChange: vi.fn() }));
vi.mock("@/hooks/useSSE", () => ({ useSSE: () => sseMock }));

function reply(attemptId: string) {
  return {
    message_id: `reply-${attemptId}`, session_id: "recovery-session", role: "assistant",
    content: "Saved answer recovered", created_at: new Date().toISOString(),
    linked_attempt_id: attemptId, tool_trail: [], metadata: { status: "completed" },
  };
}

async function mount() {
  const router = createMemoryRouter([{ path: "/", element: <Agent /> }], {
    initialEntries: ["/?session=recovery-session"],
  });
  await act(async () => { render(<RouterProvider router={router} />); });
  return router;
}

async function start(attemptId = "quiet-attempt") {
  await act(async () => {
    const store = useAgentStore.getState();
    store.startActivity(attemptId);
    store.setStatus("streaming");
  });
}

describe("chat completion recovery without SSE", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useAgentStore.getState().reset();
    apiMock.getSessionMessages.mockReset().mockResolvedValue([]);
    apiMock.getGoal.mockResolvedValue(null);
    apiMock.getLLMSettings.mockResolvedValue({
      provider: "openai-codex", model_name: "openai-codex/gpt-6.1-sol", providers: [],
      reasoning_effort: "medium", sse_timeout_seconds: 90,
    });
    Object.defineProperty(HTMLElement.prototype, "scrollTo", { configurable: true, value: vi.fn() });
  });
  afterEach(() => { cleanup(); vi.useRealTimers(); });

  it("recovers a saved answer when the completed event never arrives", async () => {
    await mount();
    await start();
    apiMock.getSessionMessages.mockResolvedValue([reply("quiet-attempt")]);
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(useAgentStore.getState().status).toBe("idle");
    expect(useAgentStore.getState().activity).toBeNull();
    expect(useAgentStore.getState().messages.some((m) => m.content === "Saved answer recovered")).toBe(true);
  });

  it("keeps reconciling after the page stopped waiting", async () => {
    await mount();
    await act(async () => {
      useAgentStore.getState().addMessage({
        type: "tool_call", content: "", timestamp: Date.now(), meta: { activity: {
          attemptId: "timed-out", state: "timeout", verb: "working", steps: [],
          startedAt: Date.now() - 90_000, endedAt: Date.now(),
        } },
      });
    });
    apiMock.getSessionMessages.mockResolvedValue([reply("timed-out")]);
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(useAgentStore.getState().messages.some((m) => m.content === "Saved answer recovered")).toBe(true);
    expect(useAgentStore.getState().messages.some((m) => m.meta?.activity?.state === "timeout")).toBe(false);
  });

  it("preserves the live answer on a failed poll and retries later", async () => {
    await mount();
    apiMock.getSessionMessages.mockRejectedValue(new Error("offline"));
    await start();
    act(() => { useAgentStore.getState().appendDelta("partial answer"); });
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(useAgentStore.getState().status).toBe("streaming");
    expect(useAgentStore.getState().streamingText).toBe("partial answer");
    apiMock.getSessionMessages.mockResolvedValue([reply("quiet-attempt")]);
    await act(async () => { await vi.advanceTimersByTimeAsync(10_000); });
    expect(useAgentStore.getState().status).toBe("idle");
  });

  it("does not clear a newer turn when an old completion check returns late", async () => {
    await mount();
    let resolve!: (messages: ReturnType<typeof reply>[]) => void;
    apiMock.getSessionMessages.mockReturnValueOnce(new Promise((r) => { resolve = r; }));
    await start("old-attempt");
    await start("new-attempt");
    await act(async () => { resolve([reply("old-attempt")]); });
    expect(useAgentStore.getState().status).toBe("streaming");
    expect(useAgentStore.getState().activity?.attemptId).toBe("new-attempt");
  });
});
