# B0 スイープをヘッドレスでまとめて実行し、CSV を保存する補助スクリプト
#
# 使い方（Release x64 をビルド済みの状態でリポジトリ直下から実行）:
# powershell -ExecutionPolicy Bypass -File scripts\run_b0_sweeps.ps1 -Exe .\build\Release\fluidsim_compare.exe
#
# Visual Studio の CMake 統合を使う場合、exe は .\out\build\x64-Release\fluidsim_compare.exe になることが多い
#

param(
    [Parameter(Mandatory=$true)]
    [string]$Exe,
    [Parameter(Mandatory=$true)]
    [string]$Stl,
    [string]$OutDir = "results_b0",
    [int]$SweepMax = 6,
    [int]$Steps = 800,
# 障害物ボクセル化で使う壁 AMR 設定
# refine=1 なら非 AMR、2-4 で見た目が変わりやすい
    [int[]]$AmrFactors = @(1,2,3),
# 粗いボクセルを固体とみなす占有率の閾値
# 0.5 が標準、0.3/0.7 で感度を確認
    [double[]]$AmrThresholds = @(0.5)
)

New-Item -ItemType Directory -Force $OutDir | Out-Null
# 単一設定のスイープを実行するヘルパー
function Run-One {
    param(
        [string]$Tag,
        [string[]]$Args
    )
    $out = Join-Path $OutDir ("b0_" + $Tag + ".csv")
    Write-Host ("[run] " + $Tag + " -> " + $out)
    & $Exe @Args | Out-File $out -Encoding ascii
}

if (-not (Test-Path $Stl)) {
    throw "STL not found: $Stl"
}
$shapeTag = [System.IO.Path]::GetFileNameWithoutExtension($Stl)

foreach($amr in $AmrFactors){
  foreach($thr in $AmrThresholds){
    $thrTag = ("thr" + ([string]$thr).Replace('.', 'p'))
    Run-One -Tag ("${shapeTag}_amr" + $amr + "_" + $thrTag) -Args @(
        "--stl", $Stl,
        "--amr-factor", "$amr",
        "--amr-threshold", "$thr",
        "--hybrid-sweep-max", "$SweepMax",
        "--hybrid-sweep-steps", "$Steps")
  }
}

Write-Host "Done. CSV files are in $OutDir"

