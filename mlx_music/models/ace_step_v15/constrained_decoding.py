"""
Constrained decoding for ACE-Step v1.5 LM planner.

Implements a Finite State Machine (FSM) logits processor that constrains
LM generation to produce valid metadata and audio codes.

Two processor classes:
- AudioCodeProcessor: Simple processor that only allows audio code tokens
- MetadataFSMProcessor: Full FSM for constrained metadata + audio codes

Note: mlx-lm passes logits as (1, vocab_size) 2D tensors. All mask
operations handle both 1D and 2D logit shapes.
"""

import logging
import re
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import mlx.core as mx
import numpy as np

from mlx_music.models.ace_step_v15.constants import (
    BPM_MAX,
    BPM_MIN,
    DURATION_MAX,
    DURATION_MIN,
    VALID_KEYSCALES,
    VALID_LANGUAGES,
    VALID_TIME_SIGNATURES,
)

logger = logging.getLogger(__name__)

NEG_INF = -1e9


class FSMState(Enum):
    """States for the metadata FSM."""

    THINK_TAG = auto()
    BPM_NAME = auto()
    BPM_VALUE = auto()
    FIELD_NEWLINE_1 = auto()
    CAPTION_NAME = auto()
    CAPTION_VALUE = auto()
    FIELD_NEWLINE_2 = auto()
    DURATION_NAME = auto()
    DURATION_VALUE = auto()
    FIELD_NEWLINE_3 = auto()
    GENRES_NAME = auto()
    GENRES_VALUE = auto()
    FIELD_NEWLINE_4 = auto()
    KEYSCALE_NAME = auto()
    KEYSCALE_VALUE = auto()
    FIELD_NEWLINE_5 = auto()
    LANGUAGE_NAME = auto()
    LANGUAGE_VALUE = auto()
    FIELD_NEWLINE_6 = auto()
    TIMESIG_NAME = auto()
    TIMESIG_VALUE = auto()
    THINK_END = auto()
    CODES = auto()
    COMPLETED = auto()


def _make_allow_mask(vocab_size: int, allowed_ids: Set[int]) -> mx.array:
    """Create a 1D additive mask: 0 for allowed tokens, NEG_INF for blocked."""
    mask_np = np.full(vocab_size, NEG_INF, dtype=np.float32)
    for tid in allowed_ids:
        if 0 <= tid < vocab_size:
            mask_np[tid] = 0.0
    return mx.array(mask_np)


def _make_block_mask(vocab_size: int, blocked_ids: Set[int]) -> mx.array:
    """Create a 1D additive mask: NEG_INF for blocked tokens, 0 for rest."""
    mask_np = np.zeros(vocab_size, dtype=np.float32)
    for tid in blocked_ids:
        if 0 <= tid < vocab_size:
            mask_np[tid] = NEG_INF
    return mx.array(mask_np)


def _apply_mask(logits: mx.array, mask: mx.array) -> mx.array:
    """Apply 1D mask to logits that may be 1D or 2D (1, vocab_size)."""
    if logits.ndim == 2:
        return logits + mask[None, :]
    return logits + mask


def _build_prefix_tree(
    tokenizer, values: List[str]
) -> Dict[Tuple[int, ...], Set[int]]:
    """Build a prefix tree mapping token ID prefixes to allowed next tokens.

    For each valid value string, encodes it and creates trie entries:
        () -> {first tokens of all values}
        (tok0,) -> {second tokens of values starting with tok0}
        ...
        complete sequence -> set containing newline token
    """
    tree: Dict[Tuple[int, ...], Set[int]] = {}
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

    for value in values:
        token_ids = tokenizer.encode(value, add_special_tokens=False)
        for i in range(len(token_ids)):
            prefix = tuple(token_ids[:i])
            tree.setdefault(prefix, set()).add(token_ids[i])
        # After complete value, allow newline
        complete = tuple(token_ids)
        tree.setdefault(complete, set()).add(newline_id)

    return tree


def _build_numeric_tree(
    tokenizer, min_val: int, max_val: int
) -> Dict[Tuple[int, ...], Set[int]]:
    """Build a prefix tree for numeric values in range [min_val, max_val]."""
    values = [str(v) for v in range(min_val, max_val + 1)]
    return _build_prefix_tree(tokenizer, values)


class AudioCodeProcessor:
    """Simple logits processor that only allows audio code tokens.

    Used for Phase 2 (codes generation) to restrict output to valid
    audio code tokens and optionally block EOS until target duration.
    """

    def __init__(self, tokenizer, target_duration: Optional[float] = None):
        self.audio_code_ids: Set[int] = set()
        self.eos_id = tokenizer.eos_token_id

        for token_str, token_id in tokenizer.get_vocab().items():
            if token_str.startswith("<|audio_code_"):
                self.audio_code_ids.add(token_id)

        self.target_codes = int(target_duration * 5) if target_duration else None
        self.codes_count = 0
        self._initialized = False
        self._last_processed = 0

        # Pre-compute allowed mask using numpy for efficiency
        vocab_size = max(tokenizer.get_vocab().values()) + 1
        allowed = set(self.audio_code_ids)
        allowed.add(self.eos_id)
        self._allowed_mask = _make_allow_mask(vocab_size, allowed)
        self._eos_block = _make_allow_mask(vocab_size, {self.eos_id})

    def set_prompt_length(self, length: int):
        # Prompt length is auto-detected on first call since mlx-lm's
        # generate_step only passes the prompt tail (1 token) to processors.
        pass

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        # Auto-detect prompt length on first call
        if not self._initialized:
            self._last_processed = tokens.shape[0]
            self._initialized = True

        # Count new audio codes since last call
        if tokens.shape[0] > self._last_processed:
            new_tokens = tokens[self._last_processed :]
            for t in new_tokens.tolist():
                if t in self.audio_code_ids:
                    self.codes_count += 1
            self._last_processed = tokens.shape[0]

        # Mask to only audio codes + EOS
        masked = _apply_mask(logits, self._allowed_mask)

        # Block EOS if haven't reached target duration
        if self.target_codes and self.codes_count < self.target_codes:
            vocab_dim = logits.shape[-1]
            if self.eos_id < vocab_dim:
                eos_penalty = mx.zeros_like(logits)
                if logits.ndim == 2:
                    eos_penalty = eos_penalty.at[0, self.eos_id].add(NEG_INF)
                else:
                    eos_penalty = eos_penalty.at[self.eos_id].add(NEG_INF)
                masked = masked + eos_penalty

        return masked


class MetadataFSMProcessor:
    """FSM-based logits processor for constrained metadata generation.

    Constrains the LM to generate valid YAML metadata inside <think> tags,
    then valid audio codes after </think>.

    The state machine enforces:
    - Fixed strings for field names ("bpm: ", "caption: ", etc.)
    - Numeric ranges for BPM (30-300), duration (10-600)
    - Valid keyscale values (70 combinations)
    - Valid language codes (51 options)
    - Valid time signatures (2, 3, 4, 6)
    - Audio code tokens only after </think>
    """

    def __init__(
        self,
        tokenizer,
        skip_caption: bool = False,
        skip_genres: bool = True,
        skip_language: bool = False,
        target_duration: Optional[float] = None,
        user_metadata: Optional[Dict[str, str]] = None,
    ):
        self.tokenizer = tokenizer
        self.skip_caption = skip_caption
        self.skip_genres = skip_genres
        self.skip_language = skip_language
        self.target_codes = int(target_duration * 5) if target_duration else None
        self.user_metadata = user_metadata or {}

        vocab = tokenizer.get_vocab()
        self.vocab_size = max(vocab.values()) + 1

        # Special tokens
        self.eos_id = tokenizer.eos_token_id
        self.think_open_id = vocab.get("<think>", None)
        self.think_close_id = vocab.get("</think>", None)
        self.newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]

        # Audio code tokens
        self.audio_code_ids: Set[int] = set()
        for token_str, token_id in vocab.items():
            if token_str.startswith("<|audio_code_"):
                self.audio_code_ids.add(token_id)

        # Pre-encode fixed strings (field names)
        self._fixed_strings = {
            FSMState.THINK_TAG: self._encode("<think>\n"),
            FSMState.BPM_NAME: self._encode("bpm: "),
            FSMState.CAPTION_NAME: self._encode("caption: "),
            FSMState.DURATION_NAME: self._encode("duration: "),
            FSMState.GENRES_NAME: self._encode("genres: "),
            FSMState.KEYSCALE_NAME: self._encode("keyscale: "),
            FSMState.LANGUAGE_NAME: self._encode("language: "),
            FSMState.TIMESIG_NAME: self._encode("timesignature: "),
            FSMState.THINK_END: self._encode("\n</think>"),
        }

        # Build prefix trees for constrained values
        self._bpm_tree = _build_numeric_tree(tokenizer, BPM_MIN, BPM_MAX)
        self._duration_tree = _build_numeric_tree(tokenizer, DURATION_MIN, DURATION_MAX)
        self._timesig_tree = _build_prefix_tree(
            tokenizer, [str(t) for t in VALID_TIME_SIGNATURES]
        )
        self._keyscale_tree = _build_prefix_tree(
            tokenizer, sorted(VALID_KEYSCALES)
        )
        self._language_tree = _build_prefix_tree(tokenizer, VALID_LANGUAGES)

        # Pre-compute audio codes mask
        audio_and_eos = set(self.audio_code_ids)
        audio_and_eos.add(self.eos_id)
        self._audio_mask = _make_allow_mask(self.vocab_size, audio_and_eos)

        # Pre-compute free text mask (block audio codes)
        self._free_text_mask = _make_block_mask(self.vocab_size, self.audio_code_ids)

        # Pre-compute newline-only mask
        self._newline_mask = _make_allow_mask(self.vocab_size, {self.newline_id})

        # Build state transitions
        self._build_transitions()

        # Runtime state
        self.state = FSMState.THINK_TAG
        self.position = 0
        self.value_tokens: List[int] = []
        self.codes_count = 0
        self._initialized = False
        self._last_processed = 0

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _build_transitions(self):
        """Build state transition map based on skip settings and user metadata."""
        self._next_state = {}

        fields = [
            ("bpm", FSMState.BPM_NAME, FSMState.BPM_VALUE, FSMState.FIELD_NEWLINE_1, False),
            ("caption", FSMState.CAPTION_NAME, FSMState.CAPTION_VALUE, FSMState.FIELD_NEWLINE_2, self.skip_caption),
            ("duration", FSMState.DURATION_NAME, FSMState.DURATION_VALUE, FSMState.FIELD_NEWLINE_3, False),
            ("genres", FSMState.GENRES_NAME, FSMState.GENRES_VALUE, FSMState.FIELD_NEWLINE_4, self.skip_genres),
            ("keyscale", FSMState.KEYSCALE_NAME, FSMState.KEYSCALE_VALUE, FSMState.FIELD_NEWLINE_5, False),
            ("language", FSMState.LANGUAGE_NAME, FSMState.LANGUAGE_VALUE, FSMState.FIELD_NEWLINE_6, self.skip_language),
            ("timesignature", FSMState.TIMESIG_NAME, FSMState.TIMESIG_VALUE, None, False),
        ]

        active_fields = []
        for field_name, name_state, value_state, nl_state, skip in fields:
            if skip or field_name in self.user_metadata:
                continue
            active_fields.append((field_name, name_state, value_state, nl_state))

        self._field_chain = active_fields

        if active_fields:
            self._next_state[FSMState.THINK_TAG] = active_fields[0][1]
        else:
            self._next_state[FSMState.THINK_TAG] = FSMState.THINK_END

        for i, (_, name_state, value_state, nl_state) in enumerate(active_fields):
            self._next_state[name_state] = value_state
            if i < len(active_fields) - 1:
                next_name = active_fields[i + 1][1]
                if nl_state:
                    self._next_state[value_state] = nl_state
                    self._next_state[nl_state] = next_name
                else:
                    self._next_state[value_state] = next_name
            else:
                self._next_state[value_state] = FSMState.THINK_END

        self._next_state[FSMState.THINK_END] = FSMState.CODES
        self._next_state[FSMState.CODES] = FSMState.COMPLETED

    def set_prompt_length(self, length: int):
        # Prompt length is auto-detected on first call since mlx-lm's
        # generate_step only passes the prompt tail (1 token) to processors.
        pass

    def reset(self):
        self.state = FSMState.THINK_TAG
        self.position = 0
        self.value_tokens = []
        self.codes_count = 0
        self._initialized = False
        self._last_processed = 0

    def _advance_state(self):
        """Advance to the next state in the FSM."""
        next_state = self._next_state.get(self.state)
        if next_state:
            self.state = next_state
            self.position = 0
            self.value_tokens = []

    def _update_state(self, token_id: int):
        """Update FSM state after a token is sampled."""
        if self.state == FSMState.COMPLETED:
            return

        if self.state == FSMState.CODES:
            if token_id in self.audio_code_ids:
                self.codes_count += 1
            elif token_id == self.eos_id:
                self._advance_state()
            return

        # Fixed string states
        if self.state in self._fixed_strings:
            self.position += 1
            if self.position >= len(self._fixed_strings[self.state]):
                self._advance_state()
            return

        # Newline states
        if self.state in (
            FSMState.FIELD_NEWLINE_1,
            FSMState.FIELD_NEWLINE_2,
            FSMState.FIELD_NEWLINE_3,
            FSMState.FIELD_NEWLINE_4,
            FSMState.FIELD_NEWLINE_5,
            FSMState.FIELD_NEWLINE_6,
        ):
            self._advance_state()
            return

        # Value states - track tokens and detect completion via newline
        if self.state in (
            FSMState.BPM_VALUE,
            FSMState.DURATION_VALUE,
            FSMState.TIMESIG_VALUE,
            FSMState.KEYSCALE_VALUE,
            FSMState.LANGUAGE_VALUE,
        ):
            if token_id == self.newline_id:
                self._advance_state()
            else:
                self.value_tokens.append(token_id)
            return

        # Caption/genres - free text terminated by newline
        if self.state in (FSMState.CAPTION_VALUE, FSMState.GENRES_VALUE):
            if token_id == self.newline_id:
                self._advance_state()
            return

    def _get_mask_for_fixed_string(self, state: FSMState) -> mx.array:
        """Get mask allowing only the next expected token in a fixed string."""
        expected = self._fixed_strings[state]
        if self.position < len(expected):
            return _make_allow_mask(self.vocab_size, {expected[self.position]})
        return mx.zeros((self.vocab_size,))

    def _get_mask_for_tree(
        self, tree: Dict[Tuple[int, ...], Set[int]]
    ) -> mx.array:
        """Get mask from prefix tree based on accumulated value tokens."""
        prefix = tuple(self.value_tokens)
        allowed = tree.get(prefix, set())

        if not allowed:
            return _make_allow_mask(self.vocab_size, {self.newline_id})

        return _make_allow_mask(self.vocab_size, allowed)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        # Auto-detect prompt length on first call
        if not self._initialized:
            self._last_processed = tokens.shape[0]
            self._initialized = True

        # Process any new tokens since last call
        if tokens.shape[0] > self._last_processed:
            new_tokens = tokens[self._last_processed :]
            for t in new_tokens.tolist():
                self._update_state(t)
            self._last_processed = tokens.shape[0]

        if self.state == FSMState.COMPLETED:
            return logits

        # Get mask for current state
        if self.state in self._fixed_strings:
            mask = self._get_mask_for_fixed_string(self.state)
        elif self.state in (
            FSMState.FIELD_NEWLINE_1,
            FSMState.FIELD_NEWLINE_2,
            FSMState.FIELD_NEWLINE_3,
            FSMState.FIELD_NEWLINE_4,
            FSMState.FIELD_NEWLINE_5,
            FSMState.FIELD_NEWLINE_6,
        ):
            mask = self._newline_mask
        elif self.state == FSMState.BPM_VALUE:
            mask = self._get_mask_for_tree(self._bpm_tree)
        elif self.state == FSMState.DURATION_VALUE:
            mask = self._get_mask_for_tree(self._duration_tree)
        elif self.state == FSMState.TIMESIG_VALUE:
            mask = self._get_mask_for_tree(self._timesig_tree)
        elif self.state == FSMState.KEYSCALE_VALUE:
            mask = self._get_mask_for_tree(self._keyscale_tree)
        elif self.state == FSMState.LANGUAGE_VALUE:
            mask = self._get_mask_for_tree(self._language_tree)
        elif self.state in (FSMState.CAPTION_VALUE, FSMState.GENRES_VALUE):
            mask = self._free_text_mask
        elif self.state == FSMState.CODES:
            result = _apply_mask(logits, self._audio_mask)
            if self.target_codes and self.codes_count < self.target_codes:
                eos_penalty = mx.zeros_like(logits)
                if logits.ndim == 2:
                    eos_penalty = eos_penalty.at[0, self.eos_id].add(NEG_INF)
                else:
                    eos_penalty = eos_penalty.at[self.eos_id].add(NEG_INF)
                result = result + eos_penalty
            return result
        else:
            return logits

        return _apply_mask(logits, mask)


def parse_lm_output(text: str) -> Tuple[Dict[str, str], str]:
    """Parse LM output into metadata dict and audio codes string.

    Args:
        text: Raw LM output containing <think>metadata</think>audio_codes

    Returns:
        (metadata_dict, audio_codes_string)
    """
    metadata = {}
    audio_codes = ""

    # Extract <think>...</think> section
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1)
        # Parse YAML-like key: value pairs
        for line in think_content.split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if key and value:
                    metadata[key] = value

    # Extract audio codes after </think>
    codes_match = re.search(r"</think>\s*(.*)", text, re.DOTALL)
    if codes_match:
        audio_codes = codes_match.group(1).strip()

    return metadata, audio_codes


def extract_audio_code_indices(
    audio_codes_str: str,
) -> List[int]:
    """Extract integer indices from audio code token string.

    Args:
        audio_codes_str: String like "<|audio_code_42|><|audio_code_1337|>..."

    Returns:
        List of integer code indices
    """
    return [
        int(m.group(1))
        for m in re.finditer(r"<\|audio_code_(\d+)\|>", audio_codes_str)
    ]


__all__ = [
    "AudioCodeProcessor",
    "MetadataFSMProcessor",
    "parse_lm_output",
    "extract_audio_code_indices",
]
