#!/bin/bash

mkdir checkpoint
cd checkpoint

echo "Downloading the harmonization model..."
wget -r "https://docs.google.com/uc?export=download&id=1wwDMRnAUp_xrTisyzZ3XTgghp15u2Xsy" -O network.pth

cd ..
echo -e "Download finished successfully!"
