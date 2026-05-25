param(
    [ValidateSet("auto", "gpu", "cpu", "cpu_parallel")]
    [string]$Mode = "auto",
    [switch]$Headless,
    [int]$Port = 8888,
    [int]$CpuWorkers = 0,
    [string]$YoloDevice = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$argsList = @("scripts\tools\start_backend.py", "--mode", $Mode, "--port", "$Port")
if ($Headless) { $argsList += "--headless" }
if ($CpuWorkers -gt 0) {
    $argsList += "--cpu-workers"
    $argsList += "$CpuWorkers"
}
if ($YoloDevice) {
    $argsList += "--yolo-device"
    $argsList += $YoloDevice
}

python @argsList
