CUDA_VISIBLE_DEVICES=6 python run.py --gene_map /import/macyang_home1/share/ychenlp/preprocessed/gene_map.tiff\
  --nuclei_mask /import/macyang_home1/share/ychenlp/preprocessed/seg_map.tiff\
  --basis /import/macyang_home1/share/ychenlp/preprocessed/basis.tiff

CUDA_VISIBLE_DEVICES=6 python test.py --gene_map /import/macyang_home1/share/ychenlp/preprocessed/gene_map.tiff\
  --nuclei_mask /import/macyang_home1/share/ychenlp/preprocessed/seg_map.tiff\
  --basis /import/macyang_home1/share/ychenlp/preprocessed/basis.tiff\
  --load_path /import/macyang_home2/ychenlp/Code/SpotFPN/Log/checkpoint/model_epoch_10.pth\
  --log_dir ./Log_test\
  --gpu 6

CUDA_VISIBLE_DEVICES=6 python run.py --gene_map /import/macyang_home1/share/ychenlp/preprocessed/gene_map.tiff\
  --nuclei_mask /import/macyang_home1/share/ychenlp/preprocessed/seg_map.tiff\
  --basis /import/macyang_home1/share/ychenlp/preprocessed/basis.tiff\
  --log_dir /import/macyang_home1/share/ychenlp/ALL_LOG/Log_train_fpn_level_0_8_um\
  --fpn_level 0\
  --gpu 6

CUDA_VISIBLE_DEVICES=6 python test.py --gene_map /import/macyang_home1/share/ychenlp/preprocessed/gene_map.tiff\
  --nuclei_mask /import/macyang_home1/share/ychenlp/preprocessed/seg_map.tiff\
  --basis /import/macyang_home1/share/ychenlp/preprocessed/basis.tiff\
  --load_path /import/macyang_home1/share/ychenlp/ALL_LOG/Log_train_fpn_level_0_8_um/checkpoint/model_epoch_10.pth\
  --log_dir /import/macyang_home1/share/ychenlp/ALL_LOG/Log_test_fpn_level_0_8_um\
  --fpn_level 0\
  --gpu 6

CUDA_VISIBLE_DEVICES=6 python test.py --gene_map /import/macyang_home1/share/ychenlp/preprocessed/gene_map.tiff\
  --nuclei_mask /import/macyang_home1/share/ychenlp/preprocessed/seg_map.tiff\
  --basis /import/macyang_home1/share/ychenlp/preprocessed/basis.tiff\
  --load_path /import/macyang_home1/share/ychenlp/ALL_LOG/Log_train_fpn_level_0_8_um/checkpoint/model_epoch_30.pth\
  --log_dir /import/macyang_home1/share/ychenlp/ALL_LOG/Log_test_fpn_level_0_8_um_30_epoch\
  --fpn_level 0\
  --gpu 6


## Ablation Study
CUDA_VISIBLE_DEVICES=6 python run.py --gene_map /import/macyang_home1/share/ychenlp/preprocessed/gene_map.tiff\
  --nuclei_mask /import/macyang_home1/share/ychenlp/preprocessed/seg_map.tiff\
  --basis /import/macyang_home1/share/ychenlp/preprocessed/basis.tiff\
  --log_dir /import/macyang_home1/share/ychenlp/ALL_LOG/Log_train_fpn_level_0_8_um_no_hiearchy\
  --fpn_level 0\
  --gpu 6