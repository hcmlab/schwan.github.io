# GPU Server Feature Extraction Setup

This repository contains Dockerfiles and pipelines to process data stored in a `Schwan_T3_Clean` structure on a remote GPU server with H100s. These containers use the latest NVIDIA NGC base images for maximum performance on modern GPUs.

## Run Instructions

You will need to mount your dataset volume when running these containers. The pipelines expect the volume to be mounted at `/data/Schwan_T3_Clean`.

**Example:**

```bash
docker build -f Dockerfile.molmo2 -t schwan_molmo2 .
docker run --gpus all -v /path/to/your/network/share/Schwan_T3_Clean:/data/Schwan_T3_Clean --env-file ../.env schwan_molmo2
```

## Annotation Viewer Specification

All feature extractors MUST output their predictions in a structured JSON format so that the Frontend Annotation Viewer can easily parse and display them.

1. **Output Directory:** All output json files must be saved under the `annotation_server` folder inside their respective session directories.
   `Schwan_T3_Clean/<Session_ID>/annotation_server/<type>_annotations.json`

2. **JSON Format:**
The JSON must be a list of objects, each containing at minimum:

```json
[
  {
    "start_time": 0.5,
    "end_time": 2.5,
    "label": "Gaze",
    "description": "Child is looking towards the caregiver.",
    "source": "molmo2",
    "confidence": 0.95
  }
]
```

Depending on your parser configuration on the frontend, `label` might map to the ICEP category, and `description` will map to the detailed tooltip definition.

## Docker Containers Provided

- **Dockerfile.molmo2:** For visual and spatial feature extraction using Molmo2.
- **Dockerfile.qwen_audio:** For Audio and Speech features using Qwen-Audio.
- **Dockerfile.qwen_omni:** For multi-modal processing using Qwen-VL/Omni.
