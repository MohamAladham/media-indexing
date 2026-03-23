#!/usr/bin/env node
/**
 * Media Indexer — Photos + Videos
 * Node.js + CLIP (transformers.js) + FFmpeg for video frames
 *
 * Setup:
 *   brew install ffmpeg
 *   npm install @xenova/transformers sharp
 *
 * Usage:
 *   node index_media.js /path/to/folder
 *   node index_media.js /path/to/folder --rebuild
 *   node index_media.js /path/to/folder --interval 10   (seconds between frames, default 5)
 */

import { AutoProcessor, CLIPVisionModelWithProjection, RawImage, env } from '@xenova/transformers';
import sharp                 from 'sharp';
import fs                    from 'fs';
import path                  from 'path';
import { execSync, spawn }   from 'child_process';
import os                    from 'os';

// ── config ────────────────────────────────────────────────────────────────────
const DB_FILE      = 'media_db.json';
const BATCH_SIZE   = 8;
const IMAGE_EXTS   = new Set(['.jpg','.jpeg','.png','.webp','.bmp','.tiff']);
const VIDEO_EXTS   = new Set(['.mp4','.mov','.avi','.mkv','.m4v','.wmv','.flv','.webm']);
const THUMB_SIZE   = 240;

env.allowLocalModels    = false;
env.cacheDir            = new URL('.cache', import.meta.url).pathname;
env.backends.onnx.wasm.numThreads = 4;

// ── arg parsing ───────────────────────────────────────────────────────────────
const args        = process.argv.slice(2);
const rebuild     = args.includes('--rebuild');
const intervalIdx = args.indexOf('--interval');
const FRAME_INTERVAL = intervalIdx >= 0 ? parseInt(args[intervalIdx + 1]) || 5 : 5;
const folder      = args.find(a => !a.startsWith('--') && isNaN(Number(a)));

if (!folder) {
  console.error('\n❌  Usage: node index_media.js /path/to/folder [--rebuild] [--interval 5]\n');
  process.exit(1);
}
const absFolder = path.resolve(folder);
if (!fs.existsSync(absFolder)) {
  console.error(`\n❌  Folder not found: ${absFolder}\n`);
  process.exit(1);
}

// ── ffmpeg check ──────────────────────────────────────────────────────────────
function checkFFmpeg() {
  try { execSync('ffmpeg -version', { stdio: 'ignore' }); return true; }
  catch { return false; }
}

// ── file discovery ────────────────────────────────────────────────────────────
function findMedia(root) {
  const images = [], videos = [];
  function walk(dir) {
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); }
    catch (e) { console.warn(`  ⚠️  Skipping ${dir}: ${e.message}`); return; }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      const ext  = path.extname(entry.name).toLowerCase();
      if (entry.isDirectory()) walk(full);
      else if (IMAGE_EXTS.has(ext)) images.push(full);
      else if (VIDEO_EXTS.has(ext)) videos.push(full);
    }
  }
  walk(root);
  return { images: images.sort(), videos: videos.sort() };
}

// ── image utilities ───────────────────────────────────────────────────────────
async function makeThumbnailB64(filePath) {
  try {
    const buf = await sharp(filePath)
      .resize(THUMB_SIZE, THUMB_SIZE, { fit: 'inside', withoutEnlargement: true })
      .jpeg({ quality: 70 }).toBuffer();
    return buf.toString('base64');
  } catch { return ''; }
}

// ── CLIP embedding ────────────────────────────────────────────────────────────
async function embedImage({ processor, visionModel }, filePath) {
  const { data, info } = await sharp(filePath)
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const image = new RawImage(new Uint8ClampedArray(data), info.width, info.height, 3);
  const inputs = await processor(image);
  const { image_embeds } = await visionModel(inputs);
  // normalize to unit vector (matches CLIP text encoder normalization)
  const arr = Array.from(image_embeds.data);
  const norm = Math.sqrt(arr.reduce((s, v) => s + v * v, 0)) + 1e-8;
  return arr.map(v => v / norm);
}

// ── video: extract frames via ffmpeg ─────────────────────────────────────────
function getVideoDuration(videoPath) {
  try {
    const out = execSync(
      `ffprobe -v error -show_entries format=duration -of csv=p=0 "${videoPath}"`,
      { encoding: 'utf8', stdio: ['pipe','pipe','ignore'] }
    );
    return parseFloat(out.trim()) || 0;
  } catch { return 0; }
}

async function extractFrame(videoPath, timestamp, tmpDir) {
  const outFile = path.join(tmpDir, `frame_${timestamp}.jpg`);
  return new Promise((resolve) => {
    const ff = spawn('ffmpeg', [
      '-ss', String(timestamp),
      '-i', videoPath,
      '-frames:v', '1',
      '-q:v', '3',
      '-y', outFile
    ], { stdio: 'ignore' });
    ff.on('close', code => resolve(code === 0 ? outFile : null));
  });
}

async function indexVideo(videoPath, extractor, tmpDir) {
  const duration = getVideoDuration(videoPath);
  if (!duration) return null;

  const timestamps = [];
  for (let t = 0; t < duration; t += FRAME_INTERVAL) timestamps.push(Math.floor(t));
  if (!timestamps.length) timestamps.push(0);

  const frames = [];
  for (const ts of timestamps) {
    const framePath = await extractFrame(videoPath, ts, tmpDir);
    if (!framePath) continue;
    try {
      const emb   = await embedImage(extractor, framePath);
      const thumb = await makeThumbnailB64(framePath);
      frames.push({ timestamp: ts, embedding: emb, thumb });
    } catch { /* skip bad frame */ }
    finally {
      try { fs.unlinkSync(framePath); } catch {}
    }
  }
  return frames.length ? frames : null;
}

// ── progress bar ──────────────────────────────────────────────────────────────
function bar(done, total, w = 28) {
  const f = Math.round(w * done / total);
  return '█'.repeat(f) + '░'.repeat(w - f);
}

// ── main ──────────────────────────────────────────────────────────────────────
async function main() {
  const hasFFmpeg = checkFFmpeg();

  console.log(`\n📁  Scanning: ${absFolder}`);
  const { images, videos } = findMedia(absFolder);
  console.log(`🖼️   Photos : ${images.length}`);
  console.log(`🎬  Videos : ${videos.length}`);
  if (!hasFFmpeg && videos.length) {
    console.log('\n⚠️   FFmpeg not found — videos will be skipped.');
    console.log('    Install with:  brew install ffmpeg\n');
  }

  // load existing DB
  let db = {};
  if (fs.existsSync(DB_FILE) && !rebuild) {
    db = JSON.parse(fs.readFileSync(DB_FILE, 'utf8'));
    console.log(`\n📂  Existing index: ${Object.keys(db).length} entries`);
  }

  const newImages = images.filter(p => !db[p]);
  const newVideos = hasFFmpeg ? videos.filter(p => !db[p]) : [];
  const total     = newImages.length + newVideos.length;

  if (!total) {
    console.log('\n✅  Everything already indexed!');
    console.log(`👉  Open media_search.html and load ${DB_FILE}\n`);
    return;
  }

  console.log(`\n⚙️   To index: ${newImages.length} photos + ${newVideos.length} videos`);
  console.log('🔄  Loading CLIP model (first run ~170 MB)…');

  const processor   = await AutoProcessor.from_pretrained('Xenova/clip-vit-base-patch32');
  const visionModel = await CLIPVisionModelWithProjection.from_pretrained(
    'Xenova/clip-vit-base-patch32', { quantized: true }
  );
  const extractor = { processor, visionModel };
  console.log('✅  Model ready\n');

  const t0 = Date.now();
  let done = 0;

  // ── index photos ────────────────────────────────────────────────────────────
  if (newImages.length) {
    console.log(`📸  Indexing photos…`);
    for (let i = 0; i < newImages.length; i += BATCH_SIZE) {
      const batch = newImages.slice(i, i + BATCH_SIZE);
      await Promise.all(batch.map(async p => {
        try {
          const emb   = await embedImage(extractor, p);
          const thumb = await makeThumbnailB64(p);
          db[p] = { type: 'photo', embedding: emb, thumb,
                    name: path.basename(p), folder: path.basename(path.dirname(p)) };
        } catch (e) {
          process.stdout.write(`\n  ⚠️  ${path.basename(p)}: ${e.message}\n`);
        }
      }));
      done += batch.length;
      process.stdout.write(`\r  [${bar(done, total)}] ${done}/${total}`);
      if (done % 200 === 0) fs.writeFileSync(DB_FILE, JSON.stringify(db));
    }
    console.log();
  }

  // ── index videos ────────────────────────────────────────────────────────────
  if (newVideos.length) {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'media-indexer-'));
    console.log(`\n🎬  Indexing videos (1 frame every ${FRAME_INTERVAL}s)…`);
    for (const vp of newVideos) {
      process.stdout.write(`\r  [${bar(done, total)}] ${done}/${total}  ${path.basename(vp).slice(0,40)}`);
      const frames = await indexVideo(vp, extractor, tmpDir);
      if (frames) {
        db[vp] = { type: 'video', frames, name: path.basename(vp),
                   folder: path.basename(path.dirname(vp)),
                   duration: getVideoDuration(vp) };
      }
      done++;
      fs.writeFileSync(DB_FILE, JSON.stringify(db)); // save after every video
    }
    fs.rmSync(tmpDir, { recursive: true, force: true });
    console.log();
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`\n⏱️   Done in ${elapsed}s`);
  fs.writeFileSync(DB_FILE, JSON.stringify(db));
  console.log(`✅  Saved → ${DB_FILE}  (${Object.keys(db).length} total entries)`);
  console.log(`\n👉  Open media_search.html in Chrome/Edge (same folder)\n`);
}

main().catch(e => { console.error('\n❌', e.message); process.exit(1); });
