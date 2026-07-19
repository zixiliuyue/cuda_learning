$server = "192.168.10.82"
$user = "sen"
$password = "123"

Write-Host "Testing SSH connection to $server..."

$testFile = [System.IO.Path]::GetTempFileName()
$testScript = @"
`$server = '$server'
`$user = '$user'
`$password = '$password'

`$psi = New-Object System.Diagnostics.ProcessStartInfo
`$psi.FileName = 'ssh'
`$psi.Arguments = "-o StrictHostKeyChecking=no -o ConnectTimeout=10 `$user@`$server `"docker --version`""
`$psi.RedirectStandardInput = `$true
`$psi.RedirectStandardOutput = `$true
`$psi.RedirectStandardError = `$true
`$psi.UseShellExecute = `$false

`$p = New-Object System.Diagnostics.Process
`$p.StartInfo = `$psi

`$p.Start() | Out-Null

Start-Sleep -Seconds 2

try {
    `$p.StandardInput.WriteLine(`$password)
    `$p.StandardInput.Flush()
} catch {
    Write-Host "Password input failed"
}

Start-Sleep -Seconds 3

if (`$p.HasExited) {
    Write-Host "Exit Code: `$(`$p.ExitCode)"
    Write-Host "STDOUT:"
    Write-Host `$p.StandardOutput.ReadToEnd()
    Write-Host "STDERR:"
    Write-Host `$p.StandardError.ReadToEnd()
} else {
    Write-Host "Process still running, killing..."
    `$p.Kill()
}
"@

$testScript | Out-File -FilePath $testFile -Encoding ASCII

& powershell.exe -ExecutionPolicy Bypass -NoProfile -File $testFile

Remove-Item $testFile -Force