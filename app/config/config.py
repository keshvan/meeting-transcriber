from dataclasses import dataclass
import os, torch

from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Config:
    hf_token: str
    device: torch.device
    diarization_model: str

def resolve_device(device_name: str) -> torch.device:
    device_name = device_name.lower()

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "DEVICE=cuda, but CUDA is not available."
            )
        return torch.device("cuda")
    
    if device_name == "cpu":
        return torch.device("cpu")
    
    raise RuntimeError(
        f"{device_name} - unknown device"
    )

config = Config(
    hf_token=os.getenv("HF_TOKEN", ""),
    device=resolve_device(os.getenv("DEVICE", "auto")),
    diarization_model=os.getenv(
        "DIARIZATION_MODEL",
        "pyannote/speaker-diarization-community-1",
    ),
)