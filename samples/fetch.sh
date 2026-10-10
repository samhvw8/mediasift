#!/usr/bin/env bash
# Test corpus: four content types, trimmed to a few minutes, <=720p.
# Files are gitignored; rerun this to rebuild. Skips what already exists.
set -euo pipefail
cd "$(dirname "$0")"

yt() {  # name id section
  [[ -f "$1.mp4" ]] && { echo "have $1"; return; }
  yt-dlp -q --no-warnings --js-runtimes node -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]" \
    --download-sections "*$3" --force-keyframes-at-cuts --merge-output-format mp4 -o "$1.%(ext)s" \
    "https://www.youtube.com/watch?v=$2"
  echo "got $1"
}

# Live action + dialogue + VFX, hard cuts (CC-BY Blender Foundation)
if [[ ! -f tears_of_steel.mp4 ]]; then
  ffmpeg -v error -nostdin -y -t 240 -i https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov \
    -c:v libx264 -preset veryfast -crf 23 -c:a aac tears_of_steel.mp4
  echo "got tears_of_steel"
fi
# Animation + music, no speech (CC-BY Blender Foundation)
yt big_buck_bunny aqz-KE-bpKQ 0:00-3:00
# Lecture: speaker, slides, live demo (CC BY-NC-SA, Harvard CS50)
yt cs50_lecture 3LPJfIKxwWc 12:00-18:00
# Fast-cut screencast montage with narration and code (Fireship; local testing only)
yt python_100s x7X9w_GIm1s 0:00-2:23

# Real sound bench: 2 clips x 20 classes of ESC-50 (CC BY-NC 3.0, Piczak), picked by bench/sense/real_sound.py's list
if [[ ! -f esc50/pick.txt ]]; then
  mkdir -p esc50
  curl -sL -o esc50/esc50.csv https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv
  python3 - <<'PY'
import csv, collections
want = ["dog","rooster","door_wood_knock","clapping","siren","church_bells","glass_breaking","keyboard_typing",
        "footsteps","laughing","crying_baby","clock_alarm","car_horn","thunderstorm","engine","mouse_click",
        "coughing","sneezing","crackling_fire","cat"]
pick = collections.defaultdict(list)
for r in csv.DictReader(open("esc50/esc50.csv")):
    if r["category"] in want and len(pick[r["category"]]) < 2 and r["fold"] in ("1", "2"):
        pick[r["category"]].append(r["filename"])
open("esc50/pick.txt", "w").write("".join(f"{c} {f}\n" for c in want for f in pick[c]))
PY
  while read -r c fn; do curl -sL -o "esc50/$fn" "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/$fn"; done < esc50/pick.txt
  echo "got esc50"
fi
# Real voice bench: six LibriSpeech dev-clean speakers (CC BY 4.0); streams the 337 MB archive once
if [[ ! -d librispeech/LibriSpeech/dev-clean/251 ]]; then
  mkdir -p librispeech
  curl -sL https://www.openslr.org/resources/12/dev-clean.tar.gz | tar -xzf - -C librispeech \
    --include='LibriSpeech/SPEAKERS.TXT' --include='LibriSpeech/dev-clean/1272/*' --include='LibriSpeech/dev-clean/1462/*' \
    --include='LibriSpeech/dev-clean/2035/*' --include='LibriSpeech/dev-clean/2078/*' --include='LibriSpeech/dev-clean/251/*' \
    --include='LibriSpeech/dev-clean/3752/*'
  echo "got librispeech"
fi

for f in *.mp4; do printf '%-22s %6.1fs\n' "$f" "$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")"; done
