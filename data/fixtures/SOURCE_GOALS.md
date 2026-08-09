# game_goals.mp4 — source and license

Real (non-synthetic) soccer game footage **containing three actual goals**. 146.4 s
(2:26), 1280x720, 29.97 fps, H.264/AAC MP4 (4385 frames), audio kept throughout.
All five segments are single-camera spectator-style footage (no broadcast
multi-camera cuts, no replays, no graphics except a small "MEHR" watermark in
segment 3 and stadium LED boards).

## Timeline of the output file

| Output time | Content | Source |
|---|---|---|
| 0:00.0 – 0:45.0 | Open play, NZ attack builds toward near goal (~0:24–0:40), corner won (~0:40) | NZ vs Canada clip 25 |
| 0:45.0 – 1:08.6 | **Goal-mouth scramble / chance** off the corner (0:45–0:51), clearance, counter | NZ vs Canada clip 26 |
| 1:08.6 – 1:30.4 | Penalty setup, **GOAL 1 at ~1:22** (Enner Valencia penalty, Ecuador v Qatar, 2022 World Cup opener), celebration to 1:30 | Valencia clip |
| 1:30.4 – 1:42.4 | Free kick swung into the box, **GOAL 2 at ~1:36** (Ajax U19 header v Inter), players wheel away celebrating | Inter–Ajax U19 clip |
| 1:42.4 – 2:26.4 | Buildup at Camp Nou, **GOAL 3 at ~2:02** (Barcelona v Malaga), home fans leap up celebrating (2:02–2:07), players walk back / restart | Barcelona clip |

Goal timestamps (verified frame-by-frame in the output): **~1:22, ~1:36, ~2:02**.
Additional clear goal-mouth chance (no goal): ~0:45–0:51.

## Sources (all Wikimedia Commons)

1. **Segments 1–2** — "2018 FIFA U-17 Women's World Cup - New Zealand vs Canada - 25.webm"
   (45.1 s) and "- 26.webm" (23.5 s), same series/author/match as `game_2min.mp4`
   (clips 20–24 were used there; 25–26 are previously unused footage).
   - https://commons.wikimedia.org/wiki/File:2018_FIFA_U-17_Women%27s_World_Cup_-_New_Zealand_vs_Canada_-_25.webm
   - https://commons.wikimedia.org/wiki/File:2018_FIFA_U-17_Women%27s_World_Cup_-_New_Zealand_vs_Canada_-_26.webm
   - Author: [NaBUru38](https://commons.wikimedia.org/wiki/User:NaBUru38) (own work) — License: **CC BY-SA 4.0**
2. **Segment 3** — "2022 FIFA World Cup's first goal by Enner Valencia of Ecuador against Qatar.webm"
   (21.8 s, 1920x1080, filmed from the stands; source: mehrnews.com)
   - https://commons.wikimedia.org/wiki/File:2022_FIFA_World_Cup%27s_first_goal_by_Enner_Valencia_of_Ecuador_against_Qatar.webm
   - Author: Mehdi Mortazavian / Mehr News Agency — License: **CC BY 4.0**
3. **Segment 4** — "Next Generation. U19. Inter vs Ajax. Ajax goal..webm"
   (16.3 s, 1280x720, pitch-level spectator; via Flickr; trimmed to first 12 s here)
   - https://commons.wikimedia.org/wiki/File:Next_Generation._U19._Inter_vs_Ajax._Ajax_goal..webm
   - Author: Flickr user credited on the file page — License: **CC BY 2.0**
4. **Segment 5** — "Barcelona goal.ogv"
   (69.3 s, 640x480 @ 11.6 fps, spectator in the stands, Camp Nou, Barcelona vs
   Malaga 2009-02-19; trimmed to first 44 s here, dropping a long crowd-pan tail)
   - https://commons.wikimedia.org/wiki/File:Barcelona_goal.ogv
   - Author: Niels Elgaard Larsen (own work) — License: **CC BY-SA 3.0**

## Attribution (use this when redistributing)

Compilation of Wikimedia Commons videos:
"2018 FIFA U-17 Women's World Cup - New Zealand vs Canada" clips 25–26 by
[NaBUru38](https://commons.wikimedia.org/wiki/User:NaBUru38), CC BY-SA 4.0;
"2022 FIFA World Cup's first goal by Enner Valencia of Ecuador against Qatar" by
Mehdi Mortazavian / Mehr News Agency, CC BY 4.0;
"Next Generation. U19. Inter vs Ajax. Ajax goal." via Flickr, CC BY 2.0;
"Barcelona goal" by Niels Elgaard Larsen, CC BY-SA 3.0.
Modified: trimmed, scaled/padded to 1280x720, transcoded, concatenated.
**The combined derivative is distributed under CC BY-SA 4.0** (ShareAlike required
by the BY-SA components; applies to this video file, not to the repository code).

## What was done (2026-07-20, ffmpeg 7.1 binary bundled by `imageio-ffmpeg`)

1. Enumerated the full NaBUru38 series via the Commons API (`list=allimages`,
   `aiprefix=2018 FIFA U-17 Women's World Cup - New Zealand vs Canada`): clips
   20–29 exist; 25–29 were downloaded and inspected frame-by-frame (1 frame / 2 s).
   **No goal was captured anywhere in that series** (the author's 10 clips cover
   only ~4.6 min of the match), so goals were sourced from other Commons videos
   (found via `list=search`, `filetype:video` + goal keywords) and each candidate
   was verified visually before use.
2. Downloaded originals with `curl -L` from `upload.wikimedia.org`.
3. Normalized each segment:
   `ffmpeg -i SRC [-t TRIM] -vf "scale=1280:720,setsar=1" -r 30000/1001 -c:v libx264
    -preset medium -crf 23 -pix_fmt yuv420p -c:a aac -b:a 128k -ar 48000 -ac 2 segN.mp4`
   (segment 5 used `scale=960:720,pad=1280:720:160:0` to pillarbox its 4:3 frame;
   segment 4 trimmed with `-t 12`, segment 5 with `-t 44`).
4. Concatenated losslessly:
   `ffmpeg -f concat -safe 0 -i concat.txt -c copy -movflags +faststart game_goals.mp4`
5. Verified with OpenCV: opens, 4385 frames @ 29.97 fps (146.4 s), first/middle/last
   frames all decode at 720x1280x3.

## Sharpness (honest assessment)

- Segments 1–2 (0:00–1:08, NZ vs Canada): daylight 720p, same camera/series as
  `game_2min.mp4` — moderately sharp, players and ball clearly trackable.
- Segment 3 (1:08–1:30, Valencia goal): the sharpest — 1080p source downscaled to
  720p, well-lit stadium; camera is high/distant so players are small; small red
  "MEHR" watermark top-left.
- Segment 4 (1:30–1:42, Ajax U19 goal): decent daylight 720p at pitch level (24 fps
  source); a spectator's head briefly occludes the lower frame right after the goal.
- Segment 5 (1:42–2:26, Barcelona goal): the softest — 640x480 @ 11.6 fps night
  footage upscaled and pillarboxed to 720p; visibly soft and choppy, but the goal
  moment and the fans erupting are unmistakable.
