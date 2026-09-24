# Start a job queue as soon as the one before it finishes, so the card is
# never idle between queues. Waits on the lock file rather than a process id,
# because run_queue clears the lock on its way out.
param(
    [Parameter(Mandatory = $true)][string]$Queue,
    [int]$TimeoutMinutes = 360
)
$repo = Split-Path -Parent $PSScriptRoot
$lock = Join-Path $repo "runs\.queue.lock"
$deadline = (Get-Date).AddMinutes($TimeoutMinutes)

while ((Test-Path $lock) -and ((Get-Date) -lt $deadline)) {
    Start-Sleep -Seconds 60
}
if (Test-Path $lock) {
    Write-Output "gave up waiting for $lock after $TimeoutMinutes minutes"
    exit 1
}
$name = [IO.Path]::GetFileNameWithoutExtension($Queue)
Write-Output "$(Get-Date -Format 'HH:mm') starting $Queue"
& (Join-Path $repo ".venv\Scripts\python.exe") -u (Join-Path $repo "scripts\run_queue.py") $Queue
