#!/bin/bash

mkdir checkpoint
cd checkpoint

echo "Downloading the harmonization model..."
wget https://drive.google.com/file/d/1wwDMRnAUp_xrTisyzZ3XTgghp15u2Xsy/view?usp=sharing -O network.pth

cd ..
echo -e "Download finished successfully!"