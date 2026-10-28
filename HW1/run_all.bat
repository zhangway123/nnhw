@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/3] data preprocess ...
python data_preprocess.py
echo [2/3] train VQ-VAE ...
python train_vqvae.py
echo [3/3] kmeans clustering + evaluation ...
python kmeans_segment.py
echo all done.
pause
