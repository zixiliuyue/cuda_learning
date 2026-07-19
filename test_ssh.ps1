$command = "which docker && docker --version"

$processInfo = New-Object System.Diagnostics.ProcessStartInfo
$processInfo.FileName = "ssh"
$processInfo.Arguments = "-o StrictHostKeyChecking=no sen@192.168.10.82 `"$command`""
$processInfo.RedirectStandardInput = $true
$processInfo.RedirectStandardOutput = $true
$processInfo.RedirectStandardError = $true
$processInfo.UseShellExecute = $false
$processInfo.CreateNoWindow = $true

$process = New-Object System.Diagnostics.Process
$process.StartInfo = $processInfo

$stdOut = ""
$stdErr = ""

$outputEvent = Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action {
    param($sender, $e)
    if ($e.Data -ne $null) {
        $script:stdOut += $e.Data + "`n"
    }
}

$errorEvent = Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action {
    param($sender, $e)
    if ($e.Data -ne $null) {
        $script:stdErr += $e.Data + "`n"
    }
}

try {
    $process.Start() | Out-Null
    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()
    
    Start-Sleep -Milliseconds 500
    
    $process.StandardInput.WriteLine("123")
    $process.StandardInput.Flush()
    
    $process.WaitForExit(10000)
    
    Start-Sleep -Milliseconds 500
    
    Unregister-Event -SourceIdentifier $outputEvent.Name
    Unregister-Event -SourceIdentifier $errorEvent.Name
    
    Write-Host "Exit Code: $($process.ExitCode)"
    Write-Host "STDOUT:"
    Write-Host $stdOut
    Write-Host "STDERR:"
    Write-Host $stdErr
}
catch {
    Write-Host "Error: $_"
}