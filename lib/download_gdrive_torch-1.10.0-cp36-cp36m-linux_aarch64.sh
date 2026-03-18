#!/bin/bash

FILE_ID="10sTElqcAVpHKL4eL1gKO-VurAI-PyXwz"
FILE_NAME="torch-1.10.0-cp36-cp36m-linux_aarch64.tar"

# Get page
curl -s -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" -o page.html

# Extract confirm + uuid
CONFIRM=$(grep -o 'confirm=[^&"]*' page.html | head -n1 | cut -d= -f2)
UUID=$(grep -o 'uuid=[^&"]*' page.html | head -n1 | cut -d= -f2)

# Download
curl -L -o "$FILE_NAME" \
"https://drive.usercontent.google.com/download?export=download&confirm=${CONFIRM}&id=${FILE_ID}&uuid=${UUID}"

rm page.html

echo "Download complete."
#chmod +x download_gdrive_torch-1.10.0-cp36-cp36m-linux_aarch64.sh
#./download_gdrive_torch-1.10.0-cp36-cp36m-linux_aarch64.sh
#tar -xvf torch-1.10.0-cp36-cp36m-linux_aarch64.tar
