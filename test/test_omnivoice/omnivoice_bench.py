#!/usr/bin/env python3
"""
OmniVoice Latency Benchmark
测试首字延迟、RTF、分句连续生成间隔性能
用法: python omnivoice_bench.py [--steps 16] [--device cuda:0] [--ref ref.wav]
"""

import argparse
import time
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

# ── 依赖检查 ──────────────────────────────────────────────────────────────────
def check_deps():
    missing = []
    for pkg in ["torch", "torchaudio", "omnivoice"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] 缺少依赖: {', '.join(missing)}")
        print("  pip install torch torchaudio omnivoice")
        sys.exit(1)

check_deps()

import torch
import torchaudio
from omnivoice import OmniVoice, OmniVoiceGenerationConfig

# ── 测试语料 ──────────────────────────────────────────────────────────────────
# Round 1: 长段话，拆分为多个短句
ROUND1_SENTENCES = [
    "今天天气不错，阳光明媚，非常适合出门散步。",
    "人工智能技术近年来发展迅猛，已经深入到我们生活的方方面面。",
    "从智能手机的语音助手，到自动驾驶汽车，再到医疗诊断辅助系统。",
    "这些技术正在以前所未有的速度改变着人类社会的面貌。",
    "未来十年，我们可以期待更多令人惊叹的创新出现在各个领域。",
]

# Round 2: 第二轮对话（间隔约 15 秒后），测试是否有热身效应消退
ROUND2_SENTENCES = [
    "语音合成技术的进步让人机交互变得更加自然流畅。",
    "实时语音对话系统对延迟有极高的要求，通常需要在三百毫秒以内完成响应。",
    "通过优化模型推理步数和批处理策略，可以显著降低端到端延迟。",
]

# ── 数据结构 ──────────────────────────────────────────────────────────────────
@dataclass
class SentenceResult:
    round_id: int
    sentence_id: int
    text: str
    text_chars: int
    time_to_first_audio_ms: float   # 首字延迟（模型 generate 耗时）
    audio_duration_s: float         # 生成音频时长
    generation_time_s: float        # generate() 总耗时
    rtf: float                      # Real-Time Factor = gen_time / audio_duration
    sample_rate: int

@dataclass
class RoundSummary:
    round_id: int
    num_sentences: int
    total_text_chars: int
    total_audio_duration_s: float
    total_generation_time_s: float
    avg_rtf: float
    avg_ttfa_ms: float              # avg time-to-first-audio
    min_ttfa_ms: float
    max_ttfa_ms: float
    avg_gen_time_s: float
    gap_before_round_s: float       # 距上一轮结束的间隔

# ── 主 Benchmark 类 ────────────────────────────────────────────────────────────
class OmniVoiceBench:
    def __init__(self, args):
        self.args = args
        self.results: List[SentenceResult] = []
        self.round_summaries: List[RoundSummary] = []
        self.model = None
        self.voice_prompt = None

    def load_model(self):
        print(f"\n{'='*60}")
        print(f"  加载 OmniVoice 模型")
        print(f"  device={self.args.device}  dtype=float16  steps={self.args.steps}")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        self.model = OmniVoice.from_pretrained(
            self.args.model,
            device_map=self.args.device,
            dtype=torch.float16,
            load_asr=(self.args.ref is not None and self.args.ref_text is None),
        )
        load_time = time.perf_counter() - t0
        print(f"  ✓ 模型加载完成，耗时 {load_time:.2f}s")

        # 预构建 voice prompt（如果有参考音频）
        if self.args.ref:
            ref_path = Path(self.args.ref)
            if not ref_path.exists():
                print(f"  [WARN] 参考音频文件不存在: {ref_path}，将使用 Auto Voice 模式")
            else:
                print(f"  ► 构建 Voice Clone Prompt: {ref_path}")
                t0 = time.perf_counter()
                self.voice_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=str(ref_path),
                    ref_text=self.args.ref_text,
                )
                prompt_time = time.perf_counter() - t0
                print(f"  ✓ Voice prompt 构建完成，耗时 {prompt_time:.2f}s")
        else:
            print(f"  ► 无参考音频，使用 Auto Voice 模式")

    def warmup(self):
        """模型热身，消除 PyTorch 编译/CUDA 初始化抖动"""
        print(f"\n► 热身推理（warmup）...")
        config = OmniVoiceGenerationConfig(
            num_step=self.args.steps,
            class_temperature=0.0,
            guidance_scale=2.0,
        )
        kw = dict(text="热身测试。", generation_config=config)
        if self.voice_prompt:
            kw["voice_clone_prompt"] = self.voice_prompt
        t0 = time.perf_counter()
        _ = self.model.generate(**kw)
        warmup_time = time.perf_counter() - t0
        print(f"  ✓ 热身完成，耗时 {warmup_time:.2f}s")

    def gen_one(self, text: str, round_id: int, sent_id: int) -> SentenceResult:
        config = OmniVoiceGenerationConfig(
            num_step=self.args.steps,
            class_temperature=0.0,      # greedy，延迟最稳定
            guidance_scale=2.0,
            preprocess_prompt=False,    # prompt 已预处理，跳过
            postprocess_output=False,   # bench 场景关掉后处理
        )
        kw = dict(text=text, generation_config=config)
        if self.voice_prompt:
            kw["voice_clone_prompt"] = self.voice_prompt

        t_start = time.perf_counter()
        audio_list = self.model.generate(**kw)
        t_end = time.perf_counter()

        gen_time = t_end - t_start
        audio_tensor = audio_list[0]        # shape (1, T)
        sr = self.model.sampling_rate
        audio_samples = audio_tensor.shape[-1]
        audio_duration = audio_samples / sr

        rtf = gen_time / audio_duration if audio_duration > 0 else float("inf")
        ttfa_ms = gen_time * 1000          # 非流式：TTFA = 总生成时间（全句输出）

        result = SentenceResult(
            round_id=round_id,
            sentence_id=sent_id,
            text=text,
            text_chars=len(text),
            time_to_first_audio_ms=ttfa_ms,
            audio_duration_s=audio_duration,
            generation_time_s=gen_time,
            rtf=rtf,
            sample_rate=sr,
        )

        # 可选保存音频
        if self.args.save_audio:
            out_dir = Path(self.args.save_audio)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"r{round_id}_s{sent_id:02d}.wav"
            torchaudio.save(str(out_path), audio_tensor, sr)

        return result

    def run_round(self, sentences: List[str], round_id: int, gap_before_s: float = 0.0) -> RoundSummary:
        print(f"\n{'─'*60}")
        print(f"  Round {round_id}  ({len(sentences)} 句)  gap_before={gap_before_s:.1f}s")
        print(f"{'─'*60}")

        round_results = []
        for i, text in enumerate(sentences):
            preview = text[:30] + ("…" if len(text) > 30 else "")
            print(f"  [{round_id}-{i+1:02d}] \"{preview}\"")
            print(f"         生成中...", end="", flush=True)

            result = self.gen_one(text, round_id, i + 1)
            round_results.append(result)
            self.results.append(result)

            print(
                f"\r         ✓ gen={result.generation_time_s*1000:.0f}ms  "
                f"audio={result.audio_duration_s:.2f}s  "
                f"RTF={result.rtf:.4f}  "
                f"chars={result.text_chars}"
            )

        # 计算本轮统计
        ttfas = [r.time_to_first_audio_ms for r in round_results]
        rtfs  = [r.rtf for r in round_results]
        summary = RoundSummary(
            round_id=round_id,
            num_sentences=len(round_results),
            total_text_chars=sum(r.text_chars for r in round_results),
            total_audio_duration_s=sum(r.audio_duration_s for r in round_results),
            total_generation_time_s=sum(r.generation_time_s for r in round_results),
            avg_rtf=sum(rtfs) / len(rtfs),
            avg_ttfa_ms=sum(ttfas) / len(ttfas),
            min_ttfa_ms=min(ttfas),
            max_ttfa_ms=max(ttfas),
            avg_gen_time_s=sum(r.generation_time_s for r in round_results) / len(round_results),
            gap_before_round_s=gap_before_s,
        )
        self.round_summaries.append(summary)
        return summary

    def print_summary(self):
        print(f"\n{'='*60}")
        print(f"  BENCHMARK SUMMARY")
        print(f"  model={self.args.model}  device={self.args.device}  steps={self.args.steps}")
        print(f"{'='*60}")

        for s in self.round_summaries:
            print(f"\n  ── Round {s.round_id} (gap_before={s.gap_before_round_s:.1f}s) ──")
            print(f"     句数:        {s.num_sentences}")
            print(f"     总字数:      {s.total_text_chars} 字")
            print(f"     总音频时长:  {s.total_audio_duration_s:.2f}s")
            print(f"     总生成耗时:  {s.total_generation_time_s:.2f}s")
            print(f"     平均 RTF:    {s.avg_rtf:.4f}  (越低越快，<1 = 超实时)")
            print(f"     平均 TTFA:   {s.avg_ttfa_ms:.0f}ms  (首字延迟)")
            print(f"     最小 TTFA:   {s.min_ttfa_ms:.0f}ms")
            print(f"     最大 TTFA:   {s.max_ttfa_ms:.0f}ms")

        print(f"\n  ── 全部句子明细 ──")
        print(f"  {'Round':>5}  {'Sent':>4}  {'Chars':>5}  {'GenTime(ms)':>11}  {'AudioDur(s)':>11}  {'RTF':>8}  文本")
        print(f"  {'-'*5}  {'-'*4}  {'-'*5}  {'-'*11}  {'-'*11}  {'-'*8}  {'-'*20}")
        for r in self.results:
            preview = r.text[:20]
            print(
                f"  {r.round_id:>5}  {r.sentence_id:>4}  {r.text_chars:>5}  "
                f"{r.time_to_first_audio_ms:>10.0f}ms  "
                f"{r.audio_duration_s:>10.2f}s  "
                f"{r.rtf:>8.4f}  {preview}"
            )

        print(f"\n  注意: OmniVoice 为非流式输出，TTFA = 整句生成时间。")
        print(f"  若要实现首字快速播放，需在应用层做切句流式化处理。")
        print(f"{'='*60}\n")

    def save_json(self, path: str):
        data = {
            "config": {
                "model": self.args.model,
                "device": self.args.device,
                "steps": self.args.steps,
                "ref_audio": self.args.ref,
            },
            "round_summaries": [asdict(s) for s in self.round_summaries],
            "sentence_results": [asdict(r) for r in self.results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 结果已保存到: {path}")

    def run(self):
        self.load_model()
        self.warmup()

        # Round 1: 长段话
        t_r1_start = time.perf_counter()
        self.run_round(ROUND1_SENTENCES, round_id=1, gap_before_s=0.0)
        t_r1_end = time.perf_counter()

        # 模拟间隔 15 秒（可配置）
        gap = self.args.gap
        print(f"\n  ⏳ 模拟对话间隔 {gap}s...")
        time.sleep(gap)
        t_r2_start = time.perf_counter()
        actual_gap = t_r2_start - t_r1_end

        # Round 2: 第二轮
        self.run_round(ROUND2_SENTENCES, round_id=2, gap_before_s=actual_gap)

        self.print_summary()

        if self.args.output:
            self.save_json(self.args.output)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="OmniVoice Latency Benchmark")
    p.add_argument("--model",      default="xtalk-bridge-service/pretrained_models/OmniVoice", help="HF repo 或本地路径")
    p.add_argument("--device",     default=None, help="cuda:0 / mps / cpu (默认自动检测)")
    p.add_argument("--steps",      type=int, default=16, help="num_step (默认16，最低延迟)")
    p.add_argument("--ref",        default=None, help="参考音频路径（不提供则 Auto Voice）")
    p.add_argument("--ref-text",   default=None, help="参考音频文本（可选，省略则走 ASR）")
    p.add_argument("--gap",        type=float, default=15.0, help="两轮之间的模拟间隔秒数")
    p.add_argument("--save-audio", default=None, help="保存生成音频的目录路径")
    p.add_argument("--output",     default="bench_results.json", help="JSON 结果输出路径")

    args = p.parse_args()

    # 自动检测 device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda:0"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
        print(f"  自动检测 device: {args.device}")

    bench = OmniVoiceBench(args)
    bench.run()


if __name__ == "__main__":
    main()