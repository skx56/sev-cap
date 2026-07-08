#!/usr/bin/env bash
# Build a varied internal eval set (30s-2min clips) from open-licensed movies
# (Blender Foundation, CC-BY) plus a Wikimedia Commons transcode.
# Any 30s-2min .mp4s you drop into clips/ work too.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p clips /tmp/sevcap_src

fetch() { # url dest
  [ -s "$2" ] && { echo "skip $(basename "$2") (exists)"; return 0; }
  echo "fetching $(basename "$2")"
  curl -fSL --retry 3 -o "$2" "$1"
}

trim() { # src start dur dest
  [ -s "$4" ] && { echo "skip $(basename "$4") (exists)"; return 0; }
  ffmpeg -hide_banner -loglevel error -ss "$2" -t "$3" -i "$1" \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 -preset veryfast -crf 26 -an -y "$4"
  echo "made $(basename "$4")"
}

BBB=/tmp/sevcap_src/bbb.mp4
ED=/tmp/sevcap_src/ed.avi
fetch "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_640x360.m4v" "$BBB" ||
  fetch "https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4" "$BBB"
fetch "https://download.blender.org/ED/ED_1024.avi" "$ED" || true

# Big Buck Bunny: slow scenic intro / character interaction / fast action / climax
trim "$BBB"  30  60 clips/bbb_intro_60s.mp4       # slow, scenic, few events
trim "$BBB" 180  90 clips/bbb_middle_90s.mp4      # character interaction
trim "$BBB" 420  45 clips/bbb_action_45s.mp4      # fast action, many cuts
trim "$BBB" 510  60 clips/bbb_climax_60s.mp4      # climax sequence

# Elephants Dream: dialogue-driven + surreal machinery scenes
if [ -s "$ED" ]; then
  trim "$ED"  60  60 clips/ed_dialogue_60s.mp4
  trim "$ED" 300  90 clips/ed_machine_90s.mp4
fi

# Wikimedia Commons real-world footage (public-domain/CC transcode)
fetch "https://upload.wikimedia.org/wikipedia/commons/transcoded/c/c0/Big_Buck_Bunny_4K.webm/Big_Buck_Bunny_4K.webm.360p.vp9.webm" \
  /tmp/sevcap_src/bbb4k.webm || true
if [ -s /tmp/sevcap_src/bbb4k.webm ]; then
  trim /tmp/sevcap_src/bbb4k.webm 60 30 clips/bbb_alt_30s.mp4
fi

echo "---"
ls -lh clips/
