docker run --privileged --gpus 'all' \
-v /home/$USER/yoonjeon_workspace/MMAR:/home/$USER/yoonjeon_workspace/MMAR \
-v /home/$USER/hdd1/yoonjeon_workspace/eval:/home/$USER/hdd1/yoonjeon_workspace/eval \
-v /home/$USER/.cache/huggingface:/root/.cache/huggingface \
-v /usr/share/fonts/truetype/:/usr/share/fonts/truetype/ \
-e HF_HOME=/root/.cache/huggingface \
-w /home/$USER/yoonjeon_workspace/MMAR/hart \
-it --shm-size=12g --rm \
--name Hart yoon313/cfg_cd:hart
