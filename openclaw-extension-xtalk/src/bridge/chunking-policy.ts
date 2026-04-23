// -----------------------------------------------------------------------
// ChunkingPolicy – sentence-level buffering before sending text to TTS.
// -----------------------------------------------------------------------

const SENTENCE_END_RE = /[。？！.?!]/;
const MIN_CHUNK_LEN = 12;
const MAX_CHUNK_LEN = 120;

export interface ChunkResult {
  ready: string[];
  remaining: string;
}

export class ChunkingPolicy {
  /**
   * Given the accumulated pending buffer, extract zero or more ready-to-send
   * chunks and return the leftover tail.
   *
   * @param skipMinLen - When true the first extracted chunk bypasses the
   *   MIN_CHUNK_LEN merge heuristic and is emitted on the very first sentence
   *   boundary.  Pass this flag for the first TTS chunk of a turn so that a
   *   short opening clause (e.g. "好的！") is dispatched immediately rather
   *   than being merged with the following sentence, which would stall TTS by
   *   one extra LLM generation cycle.
   */
  extractReady(buffer: string, skipMinLen = false): ChunkResult {
    const ready: string[] = [];
    let remaining = buffer;
    let isFirst = true;

    while (remaining.length > 0) {
      if (remaining.length >= MAX_CHUNK_LEN) {
        const sub = remaining.slice(0, MAX_CHUNK_LEN);
        const boundaryIdx = this.lastBoundary(sub);
        const splitAt = boundaryIdx !== -1 ? boundaryIdx + 1 : MAX_CHUNK_LEN;
        ready.push(remaining.slice(0, splitAt));
        remaining = remaining.slice(splitAt);
        isFirst = false;
        continue;
      }

      const boundaryIdx = this.firstBoundary(remaining);
      if (boundaryIdx === -1) break;

      const candidate = remaining.slice(0, boundaryIdx + 1);
      if (candidate.length >= MIN_CHUNK_LEN || (skipMinLen && isFirst)) {
        ready.push(candidate);
        remaining = remaining.slice(boundaryIdx + 1);
        isFirst = false;
      } else {
        const nextBoundary = this.firstBoundary(remaining.slice(boundaryIdx + 1));
        if (nextBoundary === -1) break;
        const merged = remaining.slice(0, boundaryIdx + 1 + nextBoundary + 1);
        ready.push(merged);
        remaining = remaining.slice(merged.length);
        isFirst = false;
      }
    }

    return { ready, remaining };
  }

  /** Flush whatever remains in the buffer as a final chunk (possibly below MIN). */
  flush(buffer: string): ChunkResult {
    if (buffer.trim().length === 0) return { ready: [], remaining: "" };
    return { ready: [buffer], remaining: "" };
  }

  private firstBoundary(text: string): number {
    const m = SENTENCE_END_RE.exec(text);
    return m ? m.index : -1;
  }

  private lastBoundary(text: string): number {
    let last = -1;
    let idx = 0;
    for (const ch of text) {
      if (SENTENCE_END_RE.test(ch)) last = idx;
      idx++;
    }
    return last;
  }
}
