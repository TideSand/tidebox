#!/bin/bash
# download.sh recordings|images|results [--keep]: fetch one of the Hugging Face datasets and unpack it into this checkout.
#
#   recordings  TideSand/tidebox-recordings -> data/<bench>/<recording>/<task>/              (~5 GB download)
#   images      TideSand/tidebox-images     -> images/<set>/ (archive parts, checksums, task dirs)   (~100 GB)
#   results     TideSand/tidebox-results    -> output/replay/<bench>/<recording>/<run>/        (~65 GB download, ~350 GB unpacked)
#
# The datasets hold one zstd tarball per recording, per replay run and per task-directory set; the download goes to
# .download/<name>/ (resumable: run the command again) and each tarball is unpacked next to the code, then deleted
# unless --keep. Needs the `hf` command (pip install -U huggingface_hub), tar and zstd.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TIDEBOX=$(cd "$HERE/.." && pwd)
what=${1:?usage: download.sh recordings|images|results [--keep]}; keep=${2:-}
case $what in
  recordings) dest=$TIDEBOX/data;;
  images)     dest=$TIDEBOX/images;;
  results)    dest=$TIDEBOX/output/replay;;
  *) echo "usage: download.sh recordings|images|results [--keep]"; exit 2;;
esac
command -v hf > /dev/null || { echo "the hf command is missing: pip install -U huggingface_hub"; exit 1; }
command -v zstd > /dev/null || { echo "zstd is missing: sudo apt-get install zstd"; exit 1; }
dl=$TIDEBOX/.download/$what
mkdir -p "$dl" "$dest"
hf download "TideSand/tidebox-$what" --repo-type dataset --local-dir "$dl"
cd "$dl"
# Everything but the tarballs (image archive parts, checksums, lists, the odd plain file) goes over as it is.
find . -type f ! -name '*.tar.zst' ! -path './.cache/*' ! -name README.md ! -name .gitattributes | while read -r f; do
  mkdir -p "$dest/$(dirname "$f")"; if [ -n "$keep" ]; then cp -l "$f" "$dest/$f" 2>/dev/null || cp "$f" "$dest/$f"; else mv "$f" "$dest/$f"; fi
done
# A tarball unpacks into the directory that holds it: <bench>/<recording>.tar.zst holds <bench>/<recording>/...,
# images/<set>/tasks.tar.zst holds tasks/...
find . -type f -name '*.tar.zst' | sort | while read -r t; do
  rel=$(dirname "$t"); case $what in images) into=$dest/$rel;; *) into=$dest;; esac
  echo "unpacking $t"
  mkdir -p "$into"; tar --zstd -xf "$t" -C "$into"
  [ -n "$keep" ] || rm -f "$t"
done
[ -n "$keep" ] || rm -rf "$dl"
echo "$what: in $dest"
