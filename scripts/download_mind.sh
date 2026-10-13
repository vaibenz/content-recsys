#!/usr/bin/env bash
# Downloads MIND small (about 50k users, 65k articles) into data/MINDsmall/{train,dev}.
# The dataset is released by Microsoft under the MIND license: https://msnews.github.io
# If both mirrors below fail, download MINDsmall_train.zip and MINDsmall_dev.zip from
# https://msnews.github.io manually and unzip them into the same folders.
set -euo pipefail
DEST="${1:-data/MINDsmall}"
MIRRORS=(
  "https://mind201910small.blob.core.windows.net/release"
  "https://recodatasets.z20.web.core.windows.net/newsrec"
)
for split in train dev; do
  mkdir -p "$DEST/$split"
  if [ -f "$DEST/$split/behaviors.tsv" ]; then echo "$split already present"; continue; fi
  ok=0
  for m in "${MIRRORS[@]}"; do
    if curl -fL --retry 2 -o "/tmp/MINDsmall_$split.zip" "$m/MINDsmall_$split.zip"; then ok=1; break; fi
  done
  [ "$ok" = 1 ] || { echo "Could not download $split. See the comment at the top of this script."; exit 1; }
  unzip -o -q "/tmp/MINDsmall_$split.zip" -d "$DEST/$split"
done
echo "MIND small ready in $DEST"
