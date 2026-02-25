"""Constants for ACE-Step v1.5."""

# Valid language codes (ISO 639-1 + 'yue' for Cantonese + 'unknown')
VALID_LANGUAGES = [
    "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
    "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
    "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
    "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
    "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
    "unknown",
]

# Keyscale components
KEYSCALE_NOTES = ["A", "B", "C", "D", "E", "F", "G"]
KEYSCALE_ACCIDENTALS = ["", "#", "b", "\u266f", "\u266d"]
KEYSCALE_MODES = ["major", "minor"]

# All valid keyscales: 7 notes x 5 accidentals x 2 modes = 70
VALID_KEYSCALES = frozenset(
    f"{note}{acc} {mode}"
    for note in KEYSCALE_NOTES
    for acc in KEYSCALE_ACCIDENTALS
    for mode in KEYSCALE_MODES
)

# Metadata ranges
BPM_MIN = 30
BPM_MAX = 300
DURATION_MIN = 10
DURATION_MAX = 600
VALID_TIME_SIGNATURES = [2, 3, 4, 6]

# Task types
TASK_TYPES = ["text2music", "repaint", "cover", "extract", "lego", "complete"]
TASK_TYPES_TURBO = ["text2music", "repaint", "cover"]
TASK_TYPES_BASE = ["text2music", "repaint", "cover", "extract", "lego", "complete"]

# Default instructions per task type
DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
DEFAULT_LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"
DEFAULT_LM_UNDERSTAND_INSTRUCTION = (
    "Understand the given musical conditions and describe the audio semantics accordingly:"
)
DEFAULT_LM_INSPIRED_INSTRUCTION = (
    "Expand the user's input into a more detailed and specific musical description:"
)
DEFAULT_LM_REWRITE_INSTRUCTION = (
    "Format the user's input into a more detailed and specific musical description:"
)

TASK_INSTRUCTIONS = {
    "text2music": "Fill the audio semantic mask based on the given conditions:",
    "repaint": "Repaint the mask area based on the given conditions:",
    "cover": "Generate audio semantic tokens based on the given conditions:",
    "extract": "Extract the {TRACK_NAME} track from the audio:",
    "extract_default": "Extract the track from the audio:",
    "lego": "Generate the {TRACK_NAME} track based on the audio context:",
    "lego_default": "Generate the track based on the audio context:",
    "complete": "Complete the input track with {TRACK_CLASSES}:",
    "complete_default": "Complete the input track:",
}

# Track/instrument names
TRACK_NAMES = [
    "woodwinds", "brass", "fx", "synth", "strings", "percussion",
    "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals",
]

# SFT generation prompt template
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""

# LM model memory requirements (GB)
LM_MODEL_MEMORY_GB = {
    "0.6B": 3.0,
    "1.7B": 8.0,
    "4B": 12.0,
}

# Oobleck VAE constants
VAE_SAMPLE_RATE = 48000
VAE_CHANNELS = 2  # stereo
VAE_LATENT_DIM = 64
VAE_LATENT_HZ = 25.0
VAE_DOWNSAMPLING_FACTORS = [2, 4, 4, 6, 10]  # product = 1920
VAE_COMPRESSION_RATIO = 1920  # 48000 / 25


__all__ = [
    "VALID_LANGUAGES",
    "KEYSCALE_NOTES",
    "KEYSCALE_ACCIDENTALS",
    "KEYSCALE_MODES",
    "VALID_KEYSCALES",
    "BPM_MIN",
    "BPM_MAX",
    "DURATION_MIN",
    "DURATION_MAX",
    "VALID_TIME_SIGNATURES",
    "TASK_TYPES",
    "TASK_TYPES_TURBO",
    "TASK_TYPES_BASE",
    "DEFAULT_DIT_INSTRUCTION",
    "DEFAULT_LM_INSTRUCTION",
    "DEFAULT_LM_UNDERSTAND_INSTRUCTION",
    "DEFAULT_LM_INSPIRED_INSTRUCTION",
    "DEFAULT_LM_REWRITE_INSTRUCTION",
    "TASK_INSTRUCTIONS",
    "TRACK_NAMES",
    "SFT_GEN_PROMPT",
    "LM_MODEL_MEMORY_GB",
    "VAE_SAMPLE_RATE",
    "VAE_CHANNELS",
    "VAE_LATENT_DIM",
    "VAE_LATENT_HZ",
    "VAE_DOWNSAMPLING_FACTORS",
    "VAE_COMPRESSION_RATIO",
]
