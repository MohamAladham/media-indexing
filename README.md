# Media Indexing — Local Photo & Video Search

Search your photos and videos using natural language. Fully local — no cloud, no API keys. Powered by OpenAI's CLIP model running entirely on your machine.

**Example queries:** `beach sunset`, `birthday cake`, `dog playing in snow`, `mountain hiking 2023`

---

## How it works

1. **`index_media.js`** scans a folder, generates a CLIP embedding for every photo and video frame, and saves everything to `media_db.json`.
2. **`media_search.html`** loads that file in the browser, encodes your text query with CLIP's text encoder, and ranks results by cosine similarity.

Because CLIP maps images and text into the same vector space, searching `"sunset"` finds photos of sunsets even if they were never tagged or named that way.

---

## Setup

### Requirements

- Node.js 18+
- `ffmpeg` (only needed for video indexing)

```bash
# Install ffmpeg (macOS)
brew install ffmpeg

# Install Node.js dependencies
npm install
```

### First run

The CLIP model files (~230 MB total) are cached in `.cache/` inside the project. They are downloaded automatically on first use and reused on every subsequent run.

| File | Size | Purpose |
|------|------|---------|
| `vision_model_quantized.onnx` | 85 MB | Encodes images (used by indexer) |
| `text_model_quantized.onnx` | ~40 MB | Encodes text queries (downloaded by browser on first search) |

---

## Usage

### Step 1 — Index your media

```bash
node index_media.js /path/to/your/photos
```

This scans the folder recursively, embeds every photo and video frame, and writes `media_db.json` in the current directory.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--rebuild` | off | Re-index everything, ignoring existing entries |
| `--interval N` | `5` | Seconds between video frames to index |

```bash
# Re-index everything from scratch
node index_media.js ~/Pictures --rebuild

# Index videos with a frame every 10 seconds
node index_media.js ~/Videos --interval 10
```

The indexer saves progress every 200 photos and after every video, so it is safe to interrupt and resume.

### Step 2 — Search

Open `media_search.html` in Chrome or Edge, click **Choose File**, and load your `media_db.json`.

Type any natural language query in the search box. Use the **similarity** slider to broaden or narrow results, and the **All / Photos / Videos** pills to filter by type.

Clicking a result shows a full-size preview. For videos, timestamp chips show which moments matched — click one to jump to that frame.

---

## Supported formats

| Photos | Videos |
|--------|--------|
| jpg, jpeg, png, webp, bmp, tiff | mp4, mov, avi, mkv, m4v, wmv, flv, webm |

---

## Project structure

```
photo-indexing/
├── index_media.js       # Indexer — run with Node.js
├── media_search.html    # Search UI — open in browser
├── media_db.json        # Generated index (created by indexer)
├── .cache/              # CLIP model files (persisted here)
└── package.json
```

---

## Troubleshooting

**`node index_media.js` fails immediately**
Make sure `package.json` has `"type": "module"` and you are on Node.js 18+.

**Videos are skipped**
FFmpeg is not installed or not in PATH. Install with `brew install ffmpeg`.

**Browser shows "Encoder error"**
The browser needs to download `text_model_quantized.onnx` (~40 MB) on first search. Make sure you have an internet connection for that one-time download.

**Re-running is slow**
The indexer skips already-indexed files. Use `--rebuild` only if you want to start fresh.
