#!/usr/bin/env bash
# Usage: scripts/package.sh <platform>, e.g. x86_64-linux. Packs target/release/telocroc,
# its resources and the licenses into dist/telocroc-<version>-<platform>.tar.gz.
set -euo pipefail
cd "$(dirname "$0")/.."
version=$(cargo pkgid | sed 's/.*[#@]//')
name=telocroc-$version-$1
[ -f resources/telogator-ref.fa.gz ] || scripts/fetch-resources.sh
rm -rf "dist/$name"
mkdir -p "dist/$name/resources"
cp target/release/telocroc README.md LICENSE LICENSE-TELOGATOR2 "dist/$name/"
cp resources/kmers.tsv resources/telogator-ref.fa.gz resources/telogator-ref.fa.fai \
    resources/telogator-ref-k15.txt "dist/$name/resources/"
"dist/$name/telocroc" --version
tar -C dist -czf "dist/$name.tar.gz" "$name"
rm -rf "dist/$name"
echo "dist/$name.tar.gz"
