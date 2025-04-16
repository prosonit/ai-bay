#!/bin/bash

# Exit if any command fails
set -e

# --- 1. Set variables ---
DOWNLOAD_PAGE="https://www.geekbench.com/download/linux/"
DOWNLOAD_URL=$(curl -s $DOWNLOAD_PAGE | grep -oP 'https://cdn\.geekbench\.com/Geekbench-\d+\.\d+\.\d+-Linux\.tar\.gz' | head -n 1)
ARCHIVE=$(basename "$DOWNLOAD_URL")
DIRNAME=$(basename "$ARCHIVE" .tar.gz)

# --- 2. Download Geekbench ---
echo "Downloading Geekbench from: $DOWNLOAD_URL"
wget -O "$ARCHIVE" "$DOWNLOAD_URL"

# --- 3. Extract ---
echo "Extracting $ARCHIVE ..."
tar xf "$ARCHIVE"

# --- 4. Run Geekbench ---
cd "$DIRNAME"
echo "Running Geekbench (this will take a few minutes)..."
./geekbench6

# --- 5. Cleanup option ---
echo "Do you want to delete the downloaded files? (y/N)"
read -r CLEANUP
if [[ "$CLEANUP" =~ ^[Yy]$ ]]; then
    cd ..
    rm -rf "$ARCHIVE" "$DIRNAME"
    echo "Cleaned up."
fi

