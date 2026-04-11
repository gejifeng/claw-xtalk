// -----------------------------------------------------------------------
// InterruptController – handles barge-in and user-requested interrupts.
// -----------------------------------------------------------------------
import { TurnOrchestrator } from "./turn-orchestrator";
import { SessionRegistry } from "./session-registry";

export class InterruptController {
  constructor(
    private readonly registry: SessionRegistry,
    private readonly orchestrator: TurnOrchestrator,
  ) {}

  /** Programmatic interrupt from the browser (e.g. user clicks "stop"). */
  handleUserRequested(browserSessionId: string): void {
    console.log(`[InterruptController] user-requested interrupt sid=${browserSessionId}`);
    this.orchestrator.interruptCurrentTurn(browserSessionId);
  }

  /** Voice-activity barge-in received from X-Talk sidecar. */
  handleBargeIn(browserSessionId: string, turnId: string): void {
    const mapping = this.registry.get(browserSessionId);
    if (!mapping?.currentTurn) return;
    if (mapping.currentTurn.state !== "Speaking") return;
    console.log(`[InterruptController] barge-in sid=${browserSessionId} turnId=${turnId}`);
    this.orchestrator.onBargeIn(browserSessionId, turnId);
  }
}
