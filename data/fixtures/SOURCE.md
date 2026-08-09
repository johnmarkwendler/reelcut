# game_2min.mp4 — source and license

Real (non-synthetic) soccer game footage. 120.0 s, 1280x720, 29.97 fps, H.264/AAC MP4
(3597 frames). Spectator-filmed from the stands (sideline-ish elevated view), players,
ball, referee, and pitch markings clearly visible throughout. Continuous open play —
not a broadcast feed and not an edited highlights montage.

## Source

Five consecutive raw clips of the same match, downloaded from Wikimedia Commons:

- Match: New Zealand vs Canada, third-place match, 2018 FIFA U-17 Women's World Cup,
  Estadio Charrúa, Montevideo, Uruguay (2018-12-01)
- Author: [NaBUru38](https://commons.wikimedia.org/wiki/User:NaBUru38) (own work)
- License: **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**
  <https://creativecommons.org/licenses/by-sa/4.0/>

File pages (clips 20-24 of the author's numbered series):

1. https://commons.wikimedia.org/wiki/File:2018_FIFA_U-17_Women%27s_World_Cup_-_New_Zealand_vs_Canada_-_20.webm (37.5 s)
2. https://commons.wikimedia.org/wiki/File:2018_FIFA_U-17_Women%27s_World_Cup_-_New_Zealand_vs_Canada_-_21.webm (39.0 s)
3. https://commons.wikimedia.org/wiki/File:2018_FIFA_U-17_Women%27s_World_Cup_-_New_Zealand_vs_Canada_-_22.webm (10.0 s)
4. https://commons.wikimedia.org/wiki/File:2018_FIFA_U-17_Women%27s_World_Cup_-_New_Zealand_vs_Canada_-_23.webm (14.0 s)
5. https://commons.wikimedia.org/wiki/File:2018_FIFA_U-17_Women%27s_World_Cup_-_New_Zealand_vs_Canada_-_24.webm (26.0 s)

## Attribution (use this when redistributing)

"2018 FIFA U-17 Women's World Cup - New Zealand vs Canada" clips 20-24 by
[NaBUru38](https://commons.wikimedia.org/wiki/User:NaBUru38), via Wikimedia Commons,
licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Modified: concatenated, transcoded, trimmed (see below). This derivative remains
under CC BY-SA 4.0 per the ShareAlike condition.

## What was done to it

Processed on 2026-07-20 with the ffmpeg 7.1 binary bundled by `imageio-ffmpeg`:

1. Downloaded the five original VP9/WebM clips (each 1280x720, 29.97 fps) with `curl -L`.
2. Concatenated them in numeric order (20, 21, 22, 23, 24; total 126.5 s) using the
   ffmpeg concat demuxer.
3. Re-encoded to H.264 (libx264, CRF 23, `yuv420p`) + AAC audio, kept 1280x720 @ 29.97 fps,
   `+faststart`.
4. Trimmed to the first 120 s (`-t 120`).

Command (paths abbreviated):

```
ffmpeg -f concat -safe 0 -i concat.txt -t 120 -vf "scale=1280:720,setsar=1" \
  -r 30000/1001 -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
  -c:a aac -b:a 128k -movflags +faststart game_2min.mp4
```

## Notes

- The five clips are separate consecutive recordings by the same spectator from the same
  seat, so there are four hard cuts at clip boundaries (~37.5 s, ~76.5 s, ~86.5 s, ~100.5 s).
  Within each segment the footage is continuous unedited game play.
- License nuance: the brief preferred CC0 / CC BY / public domain. No suitable clip of
  sufficient length was found under those licenses after searching Wikimedia Commons and
  archive.org; CC BY-SA 4.0 is the closest freely-licensed match. ShareAlike applies to
  redistribution of this video file (not to the code in this repository).
