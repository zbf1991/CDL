#!/usr/bin/env bash
#source activate pytorch17
#train
python main_CDL_e.py train --config-path configs/voc12.yaml

#test
#python main_CDL_e.py test \
#    --config-path configs/voc12.yaml \
#    --model-path checkpoint_vme_710.pth
#    --model-path data/models/voc12/deeplabv2_resnet101_msc/train_aug/checkpoint_final.pth



#python main_CDL_e.py crf --config-path configs/voc12.yaml

#rm -rf .sbatchlog*
#rm -rf .jobscript*
echo "finish"
