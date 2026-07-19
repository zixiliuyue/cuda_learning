$wshell = New-Object -ComObject wscript.shell
$wshell.Run('ssh -o StrictHostKeyChecking=no sen@192.168.10.82 "which docker && docker --version"')
Start-Sleep 2
$wshell.SendKeys('123{ENTER}')
Start-Sleep 5