from omnivoice import OmniVoice, OmniVoiceGenerationConfig
import torch

model = OmniVoice.from_pretrained(
    "xtalk-bridge-service/pretrained_models/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16,   # 半精度，A100 完全支持，显存减半速度更快
    load_asr=False,        # 若提供 ref_text 则不需要 Whisper，节省显存和启动时间
)

# 最低延迟配置
config = OmniVoiceGenerationConfig(
    num_step=16,           # 关键！默认32，改16约快2x，质量轻微下降但仍可用
    guidance_scale=2.0,    # 保持默认
    denoise=True,
    position_temperature=5.0,
    class_temperature=0.0, # 0 = greedy，确定性最高，延迟最稳定
    preprocess_prompt=False,  # 若参考音频已预处理，可关掉省时
    postprocess_output=False, # 流式场景可关掉后处理
)

audio = model.generate(
    text="你好，这是一段测试。",
    ref_audio="ref.wav",
    ref_text="参考音频的文字内容",  # 显式提供，跳过 Whisper ASR
    generation_config=config,
)