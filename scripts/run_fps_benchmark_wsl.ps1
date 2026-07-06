param(
  [string]$Manifest = "",
  [string]$Output = "",
  [string]$JsonOutput = "",
  [string]$Device = "cuda:0",
  [int]$ImgSz = 640,
  [int]$BatchSize = 1,
  [int]$Warmup = 30,
  [int]$Iters = 100,
  [ValidateSet("fp32", "amp", "fp16")]
  [string]$Precision = "fp32",
  [string[]]$Only = @(),
  [switch]$NoPostprocess,
  [switch]$StopOnError
)

$ErrorActionPreference = "Stop"

function Convert-ToWslPath([string]$PathValue) {
  $full = [System.IO.Path]::GetFullPath($PathValue)
  return (wsl -e wslpath -a "$full").Trim()
}

$repoWin = Split-Path -Parent $PSScriptRoot
if (-not $Manifest) {
  $Manifest = Join-Path $PSScriptRoot "fps_benchmark_manifest.json"
}
if (-not $Output) {
  $Output = Join-Path $repoWin "outputs\benchmarks\fps_results.csv"
}
if (-not $JsonOutput) {
  $JsonOutput = Join-Path $repoWin "outputs\benchmarks\fps_results.json"
}

$repoWsl = Convert-ToWslPath $repoWin
$manifestWsl = Convert-ToWslPath $Manifest
$outputWsl = Convert-ToWslPath $Output
$jsonOutputWsl = Convert-ToWslPath $JsonOutput

$onlyArgs = ""
if ($Only.Count -gt 0) {
  $quotedOnly = $Only | ForEach-Object { "'" + ($_ -replace "'", "'\''") + "'" }
  $onlyArgs = "--only " + ($quotedOnly -join " ")
}

$postprocessArg = ""
if ($NoPostprocess) {
  $postprocessArg = "--no-postprocess"
}

$stopArg = ""
if ($StopOnError) {
  $stopArg = "--stop-on-error"
}

$cmd = @"
set -eo pipefail
cd '$repoWsl'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch_2_3_0_py310
python scripts/benchmark_fps.py \
  --manifest '$manifestWsl' \
  --output '$outputWsl' \
  --json-output '$jsonOutputWsl' \
  --device '$Device' \
  --imgsz $ImgSz \
  --batch-size $BatchSize \
  --warmup $Warmup \
  --iters $Iters \
  --precision '$Precision' \
  $onlyArgs \
  $postprocessArg \
  $stopArg
"@

wsl -e bash -lc $cmd
