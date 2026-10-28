#!/usr/bin/env bash
# Downloads the subtelomere reference files (pinned to Telogator2 commit 411ed02) into
# resources/. They are large (~70 MB), so they are not kept in this repository.
set -euo pipefail
commit=411ed02
dest="$(cd "$(dirname "$0")/.." && pwd)/resources"
base="https://raw.githubusercontent.com/zstephens/telogator2/$commit/source/resources"
for f in telogator-ref.fa.gz telogator-ref.fa.fai telogator-ref-k15.txt; do
    curl -fL --retry 3 -o "$dest/$f" "$base/$f"
done
