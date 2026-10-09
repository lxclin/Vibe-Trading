import {
  applySwarmEvent,
  buildSwarmStatusFromStarted,
  buildSwarmStatusFromToolResultPreview,
} from "../swarmStatus";

describe("swarmStatus server timestamps", () => {
  it("leaves the run start unset until a server timestamp arrives", () => {
    const nowSpy = vi.spyOn(Date, "now").mockImplementation(() => {
      throw new Error("client clock must not seed swarm status");
    });

    const initial = buildSwarmStatusFromStarted({
      run_id: "run-1",
      status: "pending",
      tasks: [],
    })!;
    expect(initial.startedAt).toBe(0);

    const started = applySwarmEvent(initial, {
      type: "run_started",
      timestamp: "2026-07-29T12:00:00Z",
    });
    expect(started.startedAt).toBe(Date.parse("2026-07-29T12:00:00Z"));
    expect(nowSpy).not.toHaveBeenCalled();
  });

  it("derives agent elapsed time only from paired server event timestamps", () => {
    const initial = buildSwarmStatusFromStarted({
      run_id: "run-2",
      status: "running",
      agents: [{ id: "analyst" }],
      tasks: [{ id: "task-1", agent_id: "analyst", status: "pending" }],
    })!;
    const started = applySwarmEvent(initial, {
      type: "task_started",
      agent_id: "analyst",
      task_id: "task-1",
      timestamp: "2026-07-29T12:00:00Z",
    });
    const completed = applySwarmEvent(started, {
      type: "task_completed",
      agent_id: "analyst",
      task_id: "task-1",
      timestamp: "2026-07-29T12:00:05Z",
    });
    const missingCompletionTimestamp = applySwarmEvent(started, {
      type: "task_completed",
      agent_id: "analyst",
      task_id: "task-1",
    });

    expect(completed.agents[0].elapsed_s).toBe(5);
    expect(missingCompletionTimestamp.agents[0].elapsed_s).toBeUndefined();
  });

  it("seeds parsed and truncated previews only from serialized server times", () => {
    const parsed = buildSwarmStatusFromToolResultPreview(JSON.stringify({
      run_id: "run-3",
      preset: "demo",
      status: "completed",
      tasks: [{
        started_at: "2026-07-29T12:00:01Z",
        completed_at: "2026-07-29T12:00:08Z",
      }],
    }))!;
    const truncated = buildSwarmStatusFromToolResultPreview(
      '{"run_id":"run-4","preset":"demo","status":"completed","started_at":"2026-07-29T12:00:02Z","completed_at":"2026-07-29T12:00:09Z"',
    )!;

    expect(parsed.startedAt).toBe(Date.parse("2026-07-29T12:00:01Z"));
    expect(parsed.completedAt).toBe(Date.parse("2026-07-29T12:00:08Z"));
    expect(truncated.startedAt).toBe(Date.parse("2026-07-29T12:00:02Z"));
    expect(truncated.completedAt).toBe(Date.parse("2026-07-29T12:00:09Z"));
  });
});

describe("swarmStatus cancellation events", () => {
  it("marks a mid-flight worker cancelled, not failed, on task_cancelled", () => {
    const initial = buildSwarmStatusFromStarted({
      run_id: "run-cancel",
      status: "running",
      agents: [{ id: "analyst" }],
      tasks: [{ id: "task-1", agent_id: "analyst", status: "pending" }],
    })!;
    const started = applySwarmEvent(initial, {
      type: "task_started",
      agent_id: "analyst",
      task_id: "task-1",
      timestamp: "2026-08-22T09:00:00Z",
    });
    const cancelled = applySwarmEvent(started, {
      type: "task_cancelled",
      task_id: "task-1",
      timestamp: "2026-08-22T09:00:07Z",
      data: { iterations: 2 },
    });

    expect(cancelled.agents[0].status).toBe("cancelled");
    expect(cancelled.agents[0].iterations).toBe(2);
    expect(cancelled.agents[0].elapsed_s).toBe(7);
    expect(cancelled.agents[0].error).toBeUndefined();
  });

  it("maps the worker-side worker_cancelled event the same way", () => {
    const initial = buildSwarmStatusFromStarted({
      run_id: "run-cancel-2",
      status: "running",
      agents: [{ id: "analyst" }],
      tasks: [{ id: "task-1", agent_id: "analyst", status: "in_progress" }],
    })!;
    const cancelled = applySwarmEvent(initial, {
      type: "worker_cancelled",
      agent_id: "analyst",
      task_id: "task-1",
      data: { iterations: 1 },
    });
    expect(cancelled.agents[0].status).toBe("cancelled");
  });
});

it("tracks reuse, retry exhaustion and model waiting without a stale tool label", () => {
  let run = buildSwarmStatusFromStarted({ run_id: "r", status: "running", agents: [{ id: "analyst" }], tasks: [] })!;
  const event = (type: string, data: Record<string, unknown>) => ({ type, agent_id: "analyst", data });
  run = applySwarmEvent(run, event("tool_result", { tool: "get_financial_statements", status: "ok", cached: true }));
  expect(run.agents[0].queryState).toBe("cached");
  run = applySwarmEvent(run, event("task_heartbeat", { tool: "llm:default", phase: "llm" }));
  expect(run.agents[0].queryState).toBe("model");
  run = applySwarmEvent(run, event("tool_call", { tool: "web_search" }));
  expect(run.agents[0].queryState).toBeUndefined();
  run = applySwarmEvent(run, event("tool_result", { tool: "web_search", status: "error", retry_exhausted: true }));
  expect(run.agents[0].queryState).toBe("retry_exhausted");
  run = applySwarmEvent(run, event("worker_completed", {}));
  expect(run.agents[0].queryState).toBeUndefined();
});
