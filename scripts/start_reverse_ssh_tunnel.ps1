param(
    [string]$RemoteUser = "Administrator",
    [string]$RemoteHost = "42.51.40.136",
    [int]$RemotePort = 10022,
    [string]$LocalHost = "localhost",
    [int]$LocalPort = 22,
    [string]$LogDir = "$env:LOCALAPPDATA\ReverseSshTunnel",
    [int]$ReconnectDelaySeconds = 10
)

$ErrorActionPreference = "Stop"

function Write-TunnelLog {
    param([string]$Message)

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $script:LogFile -Value "[$timestamp] $Message" -Encoding UTF8
}

function Resolve-SshPath {
    $command = Get-Command ssh.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    $candidates = @(
        "$env:WINDIR\System32\OpenSSH\ssh.exe",
        "$env:ProgramFiles\Git\usr\bin\ssh.exe",
        "${env:ProgramFiles(x86)}\Git\usr\bin\ssh.exe",
        "$env:ProgramFiles\OpenSSH-Win64\ssh.exe"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    throw "ssh.exe not found. Install Windows OpenSSH Client or add ssh.exe to PATH."
}

New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$script:LogFile = Join-Path $LogDir "tunnel.log"
$lockFile = Join-Path $LogDir "watchdog.lock"

$mutexName = "ReverseSshTunnel_${RemoteHost}_${RemotePort}".Replace(".", "_")
$mutex = [System.Threading.Mutex]::new($false, $mutexName)
if (-not $mutex.WaitOne(0)) {
    Write-TunnelLog "Another watchdog instance is already running. Exit."
    exit 0
}

try {
    Set-Content -LiteralPath $lockFile -Value "$PID" -Encoding ASCII
    $sshExe = Resolve-SshPath
    Write-TunnelLog "Watchdog started. PID=$PID, SSH=$sshExe"
    Write-TunnelLog "Tunnel target: ${RemoteUser}@${RemoteHost}, -R ${RemotePort}:${LocalHost}:${LocalPort}"

    while ($true) {
        try {
            $reachable = Test-NetConnection -ComputerName $RemoteHost -Port 22 -InformationLevel Quiet
            if (-not $reachable) {
                Write-TunnelLog "Remote SSH is unreachable: ${RemoteHost}:22. Retry after ${ReconnectDelaySeconds}s."
                Start-Sleep -Seconds $ReconnectDelaySeconds
                continue
            }

            $arguments = @(
                "-N",
                "-T",
                "-R", "${RemotePort}:${LocalHost}:${LocalPort}",
                "-o", "ExitOnForwardFailure=yes",
                "-o", "ServerAliveInterval=30",
                "-o", "ServerAliveCountMax=3",
                "-o", "TCPKeepAlive=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "BatchMode=yes",
                "${RemoteUser}@${RemoteHost}"
            )

            Write-TunnelLog "Starting SSH tunnel."
            & $sshExe @arguments 2>&1 | ForEach-Object {
                Write-TunnelLog "ssh: $_"
            }

            $exitCode = $LASTEXITCODE
            Write-TunnelLog "SSH exited. ExitCode=$exitCode. Reconnect after ${ReconnectDelaySeconds}s."
        }
        catch {
            Write-TunnelLog "Watchdog loop error: $($_.Exception.Message). Retry after ${ReconnectDelaySeconds}s."
        }

        Start-Sleep -Seconds $ReconnectDelaySeconds
    }
}
finally {
    Remove-Item -LiteralPath $lockFile -Force -ErrorAction SilentlyContinue
    $mutex.ReleaseMutex() | Out-Null
    $mutex.Dispose()
}
