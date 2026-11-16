@echo off
setlocal
set "TIMEOUT=%~1"
if not defined TIMEOUT set "TIMEOUT=0.3"
set "BAT=%~f0"
set "MARK=CPIPE_SEARCH"
set "MARK=%MARK%_BODY"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "iex ((Get-Content -LiteralPath $env:BAT -Raw) -split $env:MARK,2)[1]"
echo.
echo %cmdcmdline% | findstr /I /C:"%~nx0" >nul && pause
exit /b
CPIPE_SEARCH_BODY
$ErrorActionPreference = 'Stop'
try { $timeoutMs = [int]([double]$env:TIMEOUT * 1000) } catch { $timeoutMs = 300 }
if ($timeoutMs -lt 200) { $timeoutMs = 200 }
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step-Ip([byte[]]$b) {
    for ($i = 3; $i -ge 0; $i--) {
        if ($b[$i] -lt 255) { $b[$i] = [byte]($b[$i] + 1); return $true }
        $b[$i] = 0
    }
    return $false
}

function Get-PrefixLength([byte[]]$mask) {
    $bits = 0
    $zero = $false
    foreach ($b in $mask) {
        for ($i = 7; $i -ge 0; $i--) {
            $on = ([int]$b -band (1 -shl $i)) -ne 0
            if ($on) {
                if ($zero) { return -1 }
                $bits++
            } else { $zero = $true }
        }
    }
    return $bits
}

function Add-SubnetHosts([string]$ip, [string]$mask, $seen, $targets) {
    if ($targets.Count -ge 1024) { return }
    $ipb = [System.Net.IPAddress]::Parse($ip).GetAddressBytes()
    if ($ipb[0] -eq 169 -and $ipb[1] -eq 254) { return }
    if ($ipb[0] -eq 127) { return }
    $mb = [System.Net.IPAddress]::Parse($mask).GetAddressBytes()
    $prefix = Get-PrefixLength $mb
    if ($prefix -lt 0) { return }
    if ($prefix -lt 22) {
        $mb = [byte[]](255, 255, 255, 0)
        $prefix = 24
    }
    if ($prefix -ge 31) {
        if (-not $seen.Contains($ip)) { [void]$seen.Add($ip); $targets.Add($ip) }
        return
    }
    $cur = New-Object byte[] 4
    $end = New-Object byte[] 4
    for ($i = 0; $i -lt 4; $i++) {
        $cur[$i] = [byte]($ipb[$i] -band $mb[$i])
        $end[$i] = [byte]($ipb[$i] -bor ($mb[$i] -bxor 255))
    }
    if (-not (Step-Ip $cur)) { return }
    while ($true) {
        $isEnd = $true
        for ($i = 0; $i -lt 4; $i++) { if ($cur[$i] -ne $end[$i]) { $isEnd = $false; break } }
        if ($isEnd) { break }
        $text = '{0}.{1}.{2}.{3}' -f $cur[0], $cur[1], $cur[2], $cur[3]
        if (-not $seen.Contains($text)) { [void]$seen.Add($text); $targets.Add($text) }
        if ($targets.Count -ge 1024) { return }
        if (-not (Step-Ip $cur)) { break }
    }
}

function Receive-Cpipe($tcp, $timeoutMs) {
    $stream = $tcp.GetStream()
    $stream.ReadTimeout = $timeoutMs
    $ms = New-Object System.IO.MemoryStream
    $buf = New-Object byte[] 2048
    while ($ms.Length -lt 65536) {
        try { $n = $stream.Read($buf, 0, $buf.Length) } catch { break }
        if ($n -le 0) { break }
        $ms.Write($buf, 0, $n)
    }
    $text = [System.Text.Encoding]::UTF8.GetString($ms.ToArray()).Trim()
    if (-not $text) { return $null }
    return $text | ConvertFrom-Json
}

$seenIp = New-Object 'System.Collections.Generic.HashSet[string]'
$targets = New-Object 'System.Collections.Generic.List[string]'
[void]$seenIp.Add('127.0.0.1')
$targets.Add('127.0.0.1')
try {
    $cfgs = @(Get-CimInstance Win32_NetworkAdapterConfiguration -Filter 'IPEnabled=True')
    $cfgs = @($cfgs | Sort-Object @{ Expression = { if ($_.DefaultIPGateway) { 0 } else { 1 } } })
    foreach ($cfg in $cfgs) {
        if (-not $cfg.IPAddress) { continue }
        for ($n = 0; $n -lt $cfg.IPAddress.Count; $n++) {
            $ip = [string]$cfg.IPAddress[$n]
            $mask = ''
            if ($cfg.IPSubnet -and $n -lt $cfg.IPSubnet.Count) { $mask = [string]$cfg.IPSubnet[$n] }
            if ($ip -notmatch '^\d+\.\d+\.\d+\.\d+$') { continue }
            if ($mask -notmatch '^\d+\.\d+\.\d+\.\d+$') { continue }
            Add-SubnetHosts $ip $mask $seenIp $targets
            if ($targets.Count -ge 1024) { break }
        }
        if ($targets.Count -ge 1024) { break }
    }
} catch {}

$seen = @{}
$found = New-Object System.Collections.Generic.List[object]
# Several cpipe on one host take 53530-53537, one port each.
$firstPort = 53530
$lastPort = 53537
$queue = New-Object 'System.Collections.Generic.Queue[object]'
foreach ($ip in $targets) { $queue.Enqueue([pscustomobject]@{ Ip = $ip; Port = $firstPort }) }
$pending = New-Object System.Collections.Generic.List[object]
while ($queue.Count -gt 0 -or $pending.Count -gt 0) {
    while ($pending.Count -lt 64 -and $queue.Count -gt 0) {
        $job = $queue.Dequeue()
        $tcp = New-Object System.Net.Sockets.TcpClient
        try { $iar = $tcp.BeginConnect($job.Ip, $job.Port, $null, $null) } catch { $tcp.Close(); continue }
        $pending.Add([pscustomobject]@{ Tcp = $tcp; Iar = $iar; Start = [DateTime]::UtcNow; Job = $job })
    }
    $next = New-Object System.Collections.Generic.List[object]
    $progress = $false
    foreach ($item in $pending) {
        $elapsed = ([DateTime]::UtcNow - $item.Start).TotalMilliseconds
        if ($item.Iar.IsCompleted) {
            $progress = $true
            try {
                $item.Tcp.EndConnect($item.Iar)
                $readMs = $timeoutMs
                if ($readMs -lt 5000) { $readMs = 5000 }
                $info = Receive-Cpipe $item.Tcp $readMs
                if ($null -ne $info -and [string]$info.name -eq 'cpipe') {
                    if ($item.Job.Port -eq $firstPort) {
                        foreach ($port in ($firstPort + 1)..$lastPort) { $queue.Enqueue([pscustomobject]@{ Ip = $item.Job.Ip; Port = $port }) }
                    }
                    $key = '{0}|{1}|{2}' -f $info.hostname, $info.insight_port, $info.device
                    $ip = [string]$info.ip
                    if ($seen.ContainsKey($key)) {
                        $oldIp = [string]$seen[$key].ip
                        if (($oldIp -eq '127.0.0.1' -or $oldIp -eq '') -and $ip -and $ip -ne '127.0.0.1') {
                            $idx = $found.IndexOf($seen[$key])
                            if ($idx -ge 0) { $found[$idx] = $info }
                            $seen[$key] = $info
                        }
                    } else {
                        $seen[$key] = $info
                        $found.Add($info)
                    }
                }
            } catch {}
            finally { $item.Tcp.Close() }
        } elseif ($elapsed -ge $timeoutMs) {
            $progress = $true
            $item.Tcp.Close()
        } else {
            $next.Add($item)
        }
    }
    $pending = $next
    if (-not $progress) { Start-Sleep -Milliseconds 20 }
}

$blocks = New-Object System.Collections.Generic.List[string]
foreach ($one in $found) {
    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($key in @('ip', 'mac', 'insight_port', 'version', 'hostname', 'device')) {
        $value = $one.$key
        if ($null -eq $value) { $value = '' }
        $lines.Add("${key}: $value")
    }
    $blocks.Add(($lines -join "`n"))
}
Write-Output ($blocks -join "`n----------------------------------`n")
