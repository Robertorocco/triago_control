#!/usr/bin/env bash
# Build the self-contained MATLAB analysis bundle (scripts + manifest, no bags).
#
# The result is ~200 KB and runs on any PC with MATLAB: no ROS, no Python, no
# path editing -- study_export_dir.m finds manifest.mat next to the scripts.
# Re-run this after every export (new manifest) or script change.
#
# Usage:  scripts/analysis/make_matlab_bundle.sh [output_dir]

set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/matlab" && pwd)"
EXPORT_DIR="${TRIAGO_STUDY_DATA_ROOT:-$HOME/exchange/triago_study_data}/matlab_export"
OUT="${1:-${TRIAGO_STUDY_DATA_ROOT:-$HOME/exchange/triago_study_data}/triago_matlab_bundle}"

[ -f "$EXPORT_DIR/manifest.mat" ] || {
    echo "no manifest.mat in $EXPORT_DIR -- run export_to_matlab.py first" >&2; exit 1; }

mkdir -p "$OUT"
cp "$SRC"/*.m "$OUT"/
cp "$EXPORT_DIR/manifest.mat" "$EXPORT_DIR/manifest.csv" "$OUT"/
cp "$SRC/../bundle_README.txt" "$OUT/README.txt"

( cd "$(dirname "$OUT")" && zip -qr "$(basename "$OUT").zip" "$(basename "$OUT")" )

echo "bundle  -> $OUT  ($(du -sh "$OUT" | cut -f1))"
echo "zip     -> $OUT.zip  ($(du -sh "$OUT.zip" | cut -f1))"
echo "trials  -> $(( $(wc -l < "$EXPORT_DIR/manifest.csv") - 1 ))"
