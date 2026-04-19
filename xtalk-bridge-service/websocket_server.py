"""Asyncio WebSocket server for the X-Talk bridge sidecar."""
import asyncio
import base64
import json
import logging

import websockets
from websockets.server import WebSocketServerProtocol

from xtalk_runtime import TTSUnavailableError

log = logging.getLogger(__name__)

# Maximum number of TTS synthesis tasks that may run concurrently.
# With a local GPU model the underlying CUDA stream serialises all kernel
# launches anyway, so a value of 2 already eliminates the Python-level
# dead-time between consecutive sentences without overloading the device.
TTS_MAX_CONCURRENT_SYNTH = 2


class _SessionState:
    def __init__(self, session_id: str, asr_engine, tts, send_json):
        self.session_id = session_id
        self.turn_id: str | None = None
        self._asr_engine = asr_engine
        self.asr_stream = None
        self.tts = tts
        self._send_json = send_json
        # text=None is the flush sentinel that marks end-of-stream.
        self.tts_queue: asyncio.Queue[tuple[str, str | None, int]] = asyncio.Queue()
        self.tts_generation = 0
        self.playback_active = False
        self.tts_seq = 0
        self.tts_worker: asyncio.Task | None = None

    async def set_turn(self, turn_id: str) -> None:
        await self._cancel_asr()
        self.turn_id = turn_id
        self.reset_tts()
        expected_turn_id = turn_id

        async def on_speech_started() -> None:
            if self.turn_id != expected_turn_id:
                return
            if not self.playback_active:
                return
            await self._send_json(
                {
                    "type": "barge_in",
                    "sessionId": self.session_id,
                    "turnId": expected_turn_id,
                }
            )

        async def on_partial(text: str) -> None:
            if self.turn_id != expected_turn_id or not text:
                return
            await self._send_json(
                {
                    "type": "asr.partial",
                    "sessionId": self.session_id,
                    "turnId": expected_turn_id,
                    "text": text,
                }
            )

        async def on_final(text: str, timing: dict | None) -> None:
            if self.turn_id != expected_turn_id or not text:
                return
            await self._send_json(
                {
                    "type": "asr.final",
                    "sessionId": self.session_id,
                    "turnId": expected_turn_id,
                    "text": text,
                    "timing": timing,
                }
            )

        async def on_error(message: str) -> None:
            if self.turn_id != expected_turn_id:
                return
            await self._send_json(
                {
                    "type": "error",
                    "turnId": expected_turn_id,
                    "message": message,
                }
            )

        try:
            self.asr_stream = self._asr_engine.create_session(
                session_id=self.session_id,
                turn_id=expected_turn_id,
                on_speech_started=on_speech_started,
                on_partial=on_partial,
                on_final=on_final,
                on_error=on_error,
            )
            await self.asr_stream.start()
        except Exception as exc:
            self.asr_stream = None
            log.exception("Failed to create ASR session session=%s turn=%s", self.session_id, turn_id)
            await self._send_json(
                {
                    "type": "error",
                    "turnId": expected_turn_id,
                    "message": f"ASR unavailable: {exc}",
                }
            )
        log.debug("[%s] turn set to %s", self.session_id, turn_id)

    async def send_audio(self, pcm_bytes: bytes) -> None:
        if self.asr_stream is None:
            return
        await self.asr_stream.send_audio(pcm_bytes)

    async def finish_audio(self) -> None:
        if self.asr_stream is None:
            return
        await self.asr_stream.finish()
        self.asr_stream = None

    async def _cancel_asr(self) -> None:
        if self.asr_stream is None:
            return
        try:
            await self.asr_stream.cancel()
        finally:
            self.asr_stream = None

    def enqueue_tts(self, turn_id: str, text: str) -> None:
        self.tts_queue.put_nowait((turn_id, text, self.tts_generation))

    def request_flush(self) -> None:
        # Put a sentinel (text=None) so the dispatch loop knows the stream is
        # complete and can emit playback.finished after all audio is sent.
        self.tts_queue.put_nowait((self.turn_id or "", None, self.tts_generation))

    def reset_tts(self) -> None:
        self.tts_generation += 1
        self.playback_active = False
        self.tts_seq = 0
        while True:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def close(self) -> None:
        await self._cancel_asr()
        self.reset_tts()
        if self.tts_worker and not self.tts_worker.done():
            self.tts_worker.cancel()
            try:
                await self.tts_worker
            except asyncio.CancelledError:
                pass


class BridgeWebSocketServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7431,
        asr_engine=None,
        tts_engine=None,
    ):
        self._host = host
        self._port = port
        self._asr_engine = asr_engine
        self._tts_engine = tts_engine

    async def start(self):
        log.info("X-Talk sidecar listening on ws://%s:%s", self._host, self._port)
        async with websockets.serve(self._handler, self._host, self._port):
            await asyncio.Future()

    async def _handler(self, ws: WebSocketServerProtocol):
        log.info("Extension connected from %s", ws.remote_address)
        sessions: dict[str, _SessionState] = {}
        pending_audio_session: str | None = None

        try:
            async for message in ws:
                if isinstance(message, bytes):
                    if pending_audio_session and pending_audio_session in sessions:
                        await sessions[pending_audio_session].send_audio(message)
                    pending_audio_session = None
                    continue

                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    log.warning("Received non-JSON text frame - ignoring")
                    continue

                msg_type = msg.get("type")
                session_id = msg.get("sessionId")

                if msg_type == "session.open":
                    sess = sessions.get(session_id)
                    if sess is None:
                        sess = _SessionState(
                            session_id,
                            self._asr_engine,
                            self._tts_engine,
                            lambda payload: self._send(ws, payload),
                        )
                        sessions[session_id] = sess
                    turn_id = msg.get("turnId")
                    if turn_id:
                        await sess.set_turn(turn_id)
                    log.info("Session opened: %s turn=%s", session_id, turn_id)
                    await self._send(ws, {"type": "session.opened", "sessionId": session_id})

                elif msg_type == "session.close":
                    sess = sessions.pop(session_id, None)
                    if sess:
                        await sess.close()
                    log.info("Session closed: %s", session_id)

                elif msg_type == "audio.frame":
                    pending_audio_session = session_id

                elif msg_type == "audio.stop":
                    sess = sessions.get(session_id)
                    if sess:
                        await sess.finish_audio()

                elif msg_type == "tts.enqueue":
                    sess = sessions.get(session_id)
                    turn_id = msg.get("turnId")
                    text = msg.get("text")
                    if not sess or not turn_id or not isinstance(text, str):
                        continue
                    if turn_id != sess.turn_id:
                        log.debug(
                            "Ignoring stale tts.enqueue session=%s turn=%s current=%s",
                            session_id,
                            turn_id,
                            sess.turn_id,
                        )
                        continue
                    sess.enqueue_tts(turn_id, text)
                    self._ensure_tts_worker(ws, sess)
                    log.debug("tts.enqueue sessionId=%s turnId=%s text=%r", session_id, turn_id, text)

                elif msg_type == "tts.flush":
                    sess = sessions.get(session_id)
                    turn_id = msg.get("turnId")
                    if not sess or not turn_id or turn_id != sess.turn_id:
                        continue
                    sess.request_flush()
                    self._ensure_tts_worker(ws, sess)
                    log.debug("tts.flush sessionId=%s turnId=%s", session_id, turn_id)

                elif msg_type == "playback.stop":
                    sess = sessions.get(session_id)
                    if not sess:
                        continue
                    sess.reset_tts()
                    log.debug("playback.stop sessionId=%s turnId=%s", session_id, msg.get("turnId"))

                else:
                    log.warning("Unknown message type: %r", msg_type)

        except websockets.exceptions.ConnectionClosed as exc:
            log.info("Extension disconnected: %s", exc)
        finally:
            for sess in sessions.values():
                await sess.close()
            sessions.clear()
            log.info("Extension connection cleaned up")

    def _ensure_tts_worker(self, ws: WebSocketServerProtocol, sess: _SessionState) -> None:
        if sess.tts_worker and not sess.tts_worker.done():
            return
        sess.tts_worker = asyncio.create_task(self._run_tts_worker(ws, sess))

    async def _run_tts_worker(self, ws: WebSocketServerProtocol, sess: _SessionState) -> None:
        """
        Pipelined TTS worker with parallel synthesis and in-order delivery.

        Architecture
        ────────────
        _dispatch (async task)
          • Pulls (turn_id, text, gen) tuples from sess.tts_queue.
          • Assigns monotonically-increasing sequence numbers.
          • Acquires a semaphore slot and spawns a _synth task for each chunk.
          • When the flush sentinel (text=None) is seen, it returns the total
            chunk count and exits.

        _synth (async tasks, up to TTS_MAX_CONCURRENT_SYNTH in parallel)
          • Calls sess.tts.synthesize(text) on a thread-pool thread.
          • Deposits the result (bytes, keyed by sequence number) into `pending`.
          • Releases its semaphore slot and fires the `ready` event.

        _send_in_order (inline, same coroutine)
          • Waits for each sequence number 0, 1, 2, … to appear in `pending`.
          • Sends tts.audio frames to the client in strict order.
          • After all chunks are delivered, sends playback.finished.

        The decoupling between synthesis and delivery means chunk N+1 can start
        synthesising as soon as a semaphore slot is free — even if chunk N is
        still being sent — while the client always receives audio in order.
        """
        generation = sess.tts_generation
        sem = asyncio.Semaphore(TTS_MAX_CONCURRENT_SYNTH)
        pending: dict[int, bytes] = {}   # seq → synthesised audio (b"" = skip)
        ready = asyncio.Event()          # fired whenever pending gains a new entry

        # ── Synthesis task ────────────────────────────────────────────────────
        async def _synth(seq: int, turn_id: str, text: str) -> None:
            try:
                if sess.tts_generation != generation or sess.turn_id != turn_id:
                    pending[seq] = b""
                    return
                audio = await asyncio.to_thread(sess.tts.synthesize, text)
                if sess.tts_generation != generation or sess.turn_id != turn_id:
                    audio = b""
                pending[seq] = audio
            except TTSUnavailableError as exc:
                log.warning("[TTS] unavailable seq=%d session=%s: %s", seq, sess.session_id, exc)
                pending[seq] = b""
            except Exception:
                log.exception("[TTS] synthesis error seq=%d session=%s", seq, sess.session_id)
                pending[seq] = b""
            finally:
                sem.release()
                ready.set()

        # ── Dispatch loop (runs as a separate task) ────────────────────────────
        async def _dispatch() -> int:
            """Consume the queue and spawn synthesis tasks.
            Returns the total number of chunks dispatched (excluding sentinel)."""
            seq = 0
            try:
                while True:
                    turn_id, text, gen = await sess.tts_queue.get()
                    try:
                        if gen != generation:
                            continue
                        if text is None:          # flush sentinel
                            ready.set()           # wake sender in case it's waiting
                            return seq
                        await sem.acquire()       # throttle concurrency
                        asyncio.create_task(_synth(seq, turn_id, text))
                        seq += 1
                    finally:
                        sess.tts_queue.task_done()
            except asyncio.CancelledError:
                ready.set()
                raise

        dispatch_task = asyncio.create_task(_dispatch())

        # ── Sender (runs in this coroutine, interleaved via await) ─────────────
        async def _send_in_order() -> None:
            send_seq = 0
            total: int | None = None

            while True:
                # Refresh total once dispatch has finished
                if total is None and dispatch_task.done():
                    try:
                        total = dispatch_task.result()
                    except Exception:
                        return

                # Exit when all chunks have been delivered
                if total is not None and send_seq >= total:
                    break

                # Wait until the next chunk is ready (race-safe check-clear-check)
                while send_seq not in pending:
                    ready.clear()
                    if send_seq in pending:        # re-check after clear
                        break
                    if sess.tts_generation != generation:
                        return
                    # Also re-check total so we don't hang after a 0-chunk flush
                    if total is None and dispatch_task.done():
                        try:
                            total = dispatch_task.result()
                        except Exception:
                            return
                        if total is not None and send_seq >= total:
                            return
                    await ready.wait()
                    if sess.tts_generation != generation:
                        return

                if sess.tts_generation != generation:
                    return

                audio = pending.pop(send_seq)
                turn_id = sess.turn_id or ""

                if audio:
                    if not sess.playback_active:
                        sess.playback_active = True
                        await self._send(ws, {
                            "type": "playback.started",
                            "sessionId": sess.session_id,
                            "turnId": turn_id,
                        })
                    sess.tts_seq += 1
                    await self._send(ws, {
                        "type": "tts.audio",
                        "sessionId": sess.session_id,
                        "turnId": turn_id,
                        "audioBase64": base64.b64encode(audio).decode("ascii"),
                        "mimeType": getattr(sess.tts, "mime_type", "audio/wav"),
                        "sampleRate": getattr(sess.tts, "sample_rate", 24000),
                        "seq": sess.tts_seq,
                    })

                send_seq += 1

            # All chunks delivered → notify client
            if sess.tts_generation == generation and sess.playback_active:
                await self._send(ws, {
                    "type": "playback.finished",
                    "sessionId": sess.session_id,
                    "turnId": sess.turn_id or "",
                })
                sess.playback_active = False

        try:
            await _send_in_order()
        except asyncio.CancelledError:
            pass
        finally:
            if not dispatch_task.done():
                dispatch_task.cancel()
                try:
                    await dispatch_task
                except (asyncio.CancelledError, Exception):
                    pass

    @staticmethod
    async def _send(ws: WebSocketServerProtocol, payload: dict):
        try:
            await ws.send(json.dumps(payload))
        except websockets.exceptions.ConnectionClosed:
            pass