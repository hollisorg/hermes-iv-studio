---
name: iv-studio-video
description: Make videos and pictures for free on this PC's own NVIDIA GPU with IV Studio (MiniMax H3 with sound and speech, the person's trained characters by name). Use when the person asks for a video, a clip, a shot, a short scene, an animated picture or a picture, and video_generate / image_generate are on the IV Studio backend.
---

# Local video and pictures with IV Studio

When `video_gen.provider` is `iv-studio`, Hermes's own `video_generate` renders on this PC through IV Studio and
ComfyUI with MiniMax H3. It's free: no API key, no credits. A render takes a few minutes, and the GPU does one at
a time.

## Before the first render

Call `iv_studio_status`. It says whether IV Studio is installed, whether its ComfyUI and GPU are ready, which
characters exist and what's missing. If IV Studio isn't installed, send the person to
https://github.com/hollisorg/iv-studio/releases/latest and stop.

## Writing the request

- **Plain words are fine.** "Tess waits at a rainy bus stop at night, sighs and says: Of course it's late." IV Studio's
  own prompt writer turns it into an H3 prompt before rendering. A prompt already in H3 form (`[Shot 1]`, `<d>…</d>`
  dialogue, `{char}`) is rendered as written.
- **Characters by name.** Name an IV Studio character (from `iv_studio_status`, e.g. Tess or Theo) and their trained
  look is used. Name one character per video.
- **Speech and sound come with the picture.** Put the spoken line in quotes after "says".
- **Length:** `duration` 1 to 15 seconds (5 is a good default).
- **Shape:** `aspect_ratio` 16:9, 9:16, 1:1, 4:3, 3:4, 3:2, 2:3 or 21:9.
- **Quality:** `resolution` 480p is the raw take (quickest, about 864x480); 720p finishes it at about 1376x768;
  1080p finishes it and upscales 2x to about 2752x1536 (slowest).
- **A picture as the first frame:** pass `image_url` (a local path or a URL); the video starts on that picture.
- **Pictures as references:** `reference_image_urls` (up to 6) puts those people or places into a new scene.
- **Same result again:** pass the `seed` the last result reported.

## Long renders

A finished video can take 3 to 7 minutes, longer than one tool call may run. If `video_generate` answers
`still_rendering` with a job id, the render is still going: tell the person it's on its way, then call
`iv_studio_result` with that job id (again if it's still rendering) until it returns the video.

## After the render

The result has the local video path, the H3 prompt IV Studio actually rendered (`h3_prompt`), the `seed`, the
character used and the render time. Send the video to the person. The render is also in IV Studio's Library, where
they can make more takes, finish at other sizes, or cut it into an edit.

## Pictures

With `image_gen.provider` set to `iv-studio`, `image_generate` makes pictures on this PC with the first picture
model IV Studio has ready (Z-Image Turbo, Krea 2, Flux 3 Klein, Qwen Image 2.1...). Passing `image_url` edits that
picture instead.

## Limits

- The person's own words decide what's made; IV Studio's MiniMax H3 terms apply (real people only with their
  consent, nothing involving minors, say it's AI when posting). IV Studio asks the person to agree once in the app.
- One render at a time on the GPU; a second request waits in IV Studio's queue.
