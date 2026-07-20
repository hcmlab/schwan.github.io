import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Python version: {sys.version}")
logger.info(f"Torch version: {torch.__version__}")
try:
    import transformers
    logger.info(f"Transformers version: {transformers.__version__}")
except ImportError:
    pass

logger.info("Attempting to load AutoProcessor...")
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B", trust_remote_code=True)
logger.info("Processor loaded successfully!")

logger.info("Attempting to load Qwen2_5_VLForConditionalGeneration...")
from transformers import Qwen2_5_VLForConditionalGeneration

try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    logger.info("Model loaded successfully with FA2!!")
except BaseException as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

logger.info("Completed without dying!")
