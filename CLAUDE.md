# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Index a folder of photos/videos
node index_media.js /path/to/folder

# Re-index from scratch
node index_media.js /path/to/folder --rebuild

# Index videos with custom frame interval (seconds)
node index_media.js /path/to/folder --interval 10
```

No build step, no test suite. The project is two files: a Node.js indexer and a standalone HTML search UI.

## Architecture

The project has two halves that share a JSON file as the interface between them.

**Indexer (`index_media.js`)** — Node.js ES module. Scans a folder recursively, runs every photo through CLIP's vision encoder, extracts frames from videos via ffmpeg and embeds those too, then writes everything to `media_db.json`. Uses `CLIPVisionModelWithProjection` + `AutoProcessor` from `@xenova/transformers` (not the `pipeline` API — that loads the full model which requires both image and text inputs). Image embeddings are L2-normalized before storing.

**Search UI (`media_search.html`)** — Single self-contained HTML file, no server needed. Loaded in the browser directly from disk. Uses `CLIPTextModelWithProjection` + `AutoTokenizer` (imported from jsDelivr CDN) to encode the user's text query, then computes cosine similarity against stored embeddings in-memory. Results are ranked and rendered as a grid with a lightbox.

**`media_db.json`** — The handoff between indexer and UI. Schema:
```json
{
  "/abs/path/to/photo.jpg": {
    "type": "photo",
    "embedding": [/* 512 floats, L2-normalized */],
    "thumb": "<base64 jpeg>",
    "name": "photo.jpg",
    "folder": "parentDirName"
  },
  "/abs/path/to/video.mp4": {
    "type": "video",
    "frames": [{ "timestamp": 0, "embedding": [...], "thumb": "<base64>" }],
    "name": "video.mp4",
    "folder": "parentDirName",
    "duration": 120
  }
}
```

## Key constraints

- `package.json` must have `"type": "module"` — the indexer uses ES module `import` syntax.
- CLIP model files live in `.cache/` (not `node_modules`). The vision model (`vision_model_quantized.onnx`, 85 MB) is used by the indexer. The text model (`text_model_quantized.onnx`, ~40 MB) is downloaded by the browser on first search and cached in browser storage. `env.cacheDir` in `index_media.js` points to `.cache/` relative to the script file.
- The indexer uses `CLIPVisionModelWithProjection` (not `pipeline('feature-extraction', ...)`). The `feature-extraction` pipeline loads the full combined CLIP ONNX which requires both `pixel_values` and `input_ids` — it cannot be used for image-only or text-only embedding.
- Image pixels must go through `sharp` → `RawImage` before passing to the processor. Do not manually normalize or resize — the `AutoProcessor` handles CLIP's preprocessing (resize to 224×224, normalize, CHW conversion).
- Text and image embeddings must both be L2-normalized to make cosine similarity work correctly across modalities.
