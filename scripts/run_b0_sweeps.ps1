# summary: B0 sweep をまとめて実行する補助スクリプト
# 
# Windows PowerShell helper to run the B0 sweep in headless mode
# and save CSV outputs.
#
# Usage (from repo root, after building Release x64):
# powershell -ExecutionPolicy Bypass -File scripts\run_b0_sweeps.ps1 -Exe .\build\Release\fluidsim_compare.exe
#
# If you use Visual Studio CMake integration, the exe path is often:
# .\out\build\x64-Release\fluidsim_compare.exe
# 

param(
    [Parameter(Mandatory=$true)]
    [string]$Exe,
    [string]$OutDir = "results_b0",
    [int]$SweepMax = 6,
    [int]$Steps = 800,
    # "wall AMR" settings used by the obstacle voxelizer.
    # refine=1 means no supersampling; 2..4 usually gives a visible difference.
    [int[]]$AmrFactors = @(1,2,3),
    # Coverage threshold to mark a coarse voxel solid.
    # 0.5 is the default; try 0.3 and 0.7 to see sensitivity.
    [double[]]$AmrThresholds = @(0.5)
)

New-Item -ItemType Directory -Force $OutDir | Out-Null

# summary: Run-One の処理を行う
# param: なし
# return: なし
function Run-One {
    param(
        [string]$Tag,
        [string[]]$Args
    )
    $out = Join-Path $OutDir ("b0_" + $Tag + ".csv")
    Write-Host ("[run] " + $Tag + " -> " + $out)
    & $Exe @Args | Out-File $out -Encoding ascii
}

# Procedural shapes (no external files):
foreach($amr in $AmrFactors){
  foreach($thr in $AmrThresholds){
    $thrTag = ("thr" + ([string]$thr).Replace('.', 'p'))
    Run-One -Tag ("teardrop_amr" + $amr + "_" + $thrTag) -Args @(
        "--teardrop",
        "--amr-factor", "$amr",
        "--amr-threshold", "$thr",
        "--hybrid-sweep-max", "$SweepMax",
        "--hybrid-sweep-steps", "$Steps")

    Run-One -Tag ("fan_amr" + $amr + "_" + $thrTag)      -Args @(
        "--fan",
        "--amr-factor", "$amr",
        "--amr-threshold", "$thr",
        "--hybrid-sweep-max", "$SweepMax",
        "--hybrid-sweep-steps", "$Steps")
  }
}

# STL example (edit the path):
# $stl = ".\assets\your_model.stl"
# Run-One -Tag ("stl_amr2") -Args @("--stl", $stl, "--amr-factor", "2", "--hybrid-sweep-max", "$SweepMax", "--hybrid-sweep-steps", "$Steps")

Write-Host "Done. CSV files are in $OutDir"
