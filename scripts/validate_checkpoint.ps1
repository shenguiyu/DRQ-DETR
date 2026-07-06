param(
  [Parameter(Mandatory=$true)][string]$Checkpoint,
  [Parameter(Mandatory=$true)][string]$ValImages,
  [Parameter(Mandatory=$true)][string]$ValJson,
  [string]$Config = "configs/experiments/visdrone2019/drq_detr.yml",
  [string]$OutputDir = "./outputs/validation/visdrone2019"
)

python train.py `
  -c $Config `
  -r $Checkpoint `
  --test-only `
  -u val_dataloader.dataset.img_folder=$ValImages `
     val_dataloader.dataset.ann_file=$ValJson `
     output_dir=$OutputDir
