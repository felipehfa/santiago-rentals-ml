param([string]$svg, [string]$png, [int]$ancho = 2300, [int]$alto = 900)
$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
$url = 'file:///' + $svg.Replace('\','/').Replace(' ','%20')
& $edge --headless --disable-gpu "--screenshot=$png" "--window-size=$ancho,$alto" $url 2>&1 | Out-Null
Test-Path $png
