param(
    [string]$TaskName = "ReverseSshTunnel",
    [string]$WatchdogScript = "$PSScriptRoot\start_reverse_ssh_tunnel.ps1",
    [string]$InstallDir = "$env:LOCALAPPDATA\ReverseSshTunnel",
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"

$resolvedWatchdog = Resolve-Path -LiteralPath $WatchdogScript -ErrorAction Stop
$installedWatchdog = Join-Path $InstallDir "start_reverse_ssh_tunnel.ps1"
$logFile = Join-Path $InstallDir "tunnel.log"
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
Copy-Item -LiteralPath $resolvedWatchdog -Destination $installedWatchdog -Force

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$installedWatchdog`""

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Days 3650) `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1)

$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force | Out-Null

if (-not $NoStart) {
    Start-ScheduledTask -TaskName $TaskName
}

Write-Host "已注册计划任务: $TaskName"
Write-Host "运行用户: $currentUser"
Write-Host "守护脚本: $installedWatchdog"
Write-Host "日志文件: $logFile"
Write-Host "查看状态: Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "查看日志: Get-Content `"$logFile`" -Tail 80"
