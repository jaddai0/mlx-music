"""Prompt formatting and delay pattern for MOSS-SoundEffect (MLX)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from .config import MossSoundEffectConfig


DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}"
    "{% else %}{% for content in message['content'] %}"
    "{% if content.get('type') == 'text' %}{{ content['text'] }}{% endif %}"
    "{% endfor %}{% endif %}<|im_end|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def apply_delay_pattern(codes: mx.array, n_vq: int, pad_code: int = 1024) -> mx.array:
    """Apply delay pattern to audio codes.

    Staggers codebook channels so channel i is delayed by i time steps.

    Args:
        codes: (T, n_vq) audio codes.
        n_vq: Number of VQ channels.
        pad_code: Padding value.

    Returns:
        delayed: (T + n_vq - 1, n_vq) delayed codes.

    Example (T=2, n_vq=3):
        Input:  [[a0, b0, c0], [a1, b1, c1]]
        Output: [[a0,  P,  P],
                 [a1, b0,  P],
                 [ P, b1, c0],
                 [ P,  P, c1]]
    """
    T = codes.shape[0]
    out_len = T + n_vq - 1
    columns = []
    for i in range(n_vq):
        prefix = mx.full((i,), pad_code, dtype=codes.dtype)
        suffix_len = out_len - i - T
        suffix = mx.full((suffix_len,), pad_code, dtype=codes.dtype)
        columns.append(mx.concatenate([prefix, codes[:, i], suffix], axis=0))

    return mx.stack(columns, axis=1)


def apply_de_delay_pattern(delay_codes: mx.array, n_vq: int, pad_code: int = 1024) -> mx.array:
    """Remove delay pattern from audio codes (inverse of apply_delay_pattern).

    Args:
        delay_codes: (T', n_vq) delayed codes.
        n_vq: Number of VQ channels.
        pad_code: Padding value.

    Returns:
        codes: (T, n_vq) original codes where T = T' - n_vq + 1.
    """
    T_prime = delay_codes.shape[0]
    T = T_prime - n_vq + 1
    if T <= 0:
        return mx.full((0, n_vq), pad_code, dtype=delay_codes.dtype)

    columns = [delay_codes[i:i + T, i] for i in range(n_vq)]
    return mx.stack(columns, axis=1)


@dataclass
class UserMessage:
    """User message for MOSS-SoundEffect prompt."""
    text: str | None = None
    tokens: int | None = None
    quality: str | None = None
    sound_event: str | None = None
    ambient_sound: str | None = None
    instruction: str | None = None
    language: str | None = None

    def format_content(self) -> str:
        template = """<user_inst>
- Reference(s):
None
- Instruction:
{instruction}
- Tokens:
{tokens}
- Quality:
{quality}
- Sound Event:
{sound_event}
- Ambient Sound:
{ambient_sound}
- Language:
{language}
- Text:
{text}
</user_inst>"""
        return (
            template
            .replace("{instruction}", str(self.instruction))
            .replace("{tokens}", str(self.tokens))
            .replace("{quality}", str(self.quality))
            .replace("{sound_event}", str(self.sound_event))
            .replace("{ambient_sound}", str(self.ambient_sound))
            .replace("{language}", str(self.language))
            .replace("{text}", str(self.text))
        )


def render_chat_template(messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
    """Fallback renderer matching the source chat_template.jinja."""
    parts: list[str] = []
    for message in messages:
        content = message["content"]
        if not isinstance(content, str):
            content = str(content)
        parts.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def build_prompt(
    tokenizer: Any,
    config: MossSoundEffectConfig,
    prompt: str,
    duration_seconds: float = 5.0,
    quality: str = "high",
) -> mx.array:
    """Build input_ids tensor for sound effect generation.

    Args:
        tokenizer: HuggingFace tokenizer.
        config: Model config.
        prompt: Text description of the sound effect.
        duration_seconds: Target duration in seconds.
        quality: Audio quality descriptor.

    Returns:
        input_ids: (1, S, 1 + n_vq) input tensor ready for generation.
    """
    # Calculate target tokens from duration
    # 24000 Hz / 1920 downsample = 12.5 tokens/sec
    tokens_per_sec = 12.5
    target_tokens = int(duration_seconds * tokens_per_sec)

    msg = UserMessage(
        text=None,
        tokens=target_tokens,
        quality=quality,
        sound_event=prompt,
        ambient_sound=None,
        instruction=None,
        language=None,
    )

    content = msg.format_content()

    # Build chat template
    messages = [{"role": "user", "content": content}]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = render_chat_template(messages, add_generation_prompt=True)

    # Tokenize
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Build (S, 1 + n_vq) tensor: text tokens + pad codes for audio channels
    seq_len = len(token_ids)
    text_ids = mx.array(token_ids, dtype=mx.int32)[:, None]
    audio_pad = mx.full((seq_len, config.n_vq), config.audio_pad_code, dtype=mx.int32)
    input_ids = mx.concatenate([text_ids, audio_pad], axis=1)

    # Add batch dimension
    return input_ids[None, :, :]  # (1, S, 1 + n_vq)
