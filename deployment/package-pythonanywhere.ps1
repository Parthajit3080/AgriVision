$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$packagePath = Join-Path $PSScriptRoot "AgriVision-free-preview.zip"
$stagingPath = Join-Path $PSScriptRoot ".pythonanywhere-staging"
$resolvedDeployment = [System.IO.Path]::GetFullPath($PSScriptRoot)
$resolvedStaging = [System.IO.Path]::GetFullPath($stagingPath)

if (-not $resolvedStaging.StartsWith($resolvedDeployment + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write outside the deployment folder."
}

if (-not (Test-Path (Join-Path $projectRoot "out\index.html"))) {
    throw "Build the static site first with pnpm run build."
}

if (Test-Path $stagingPath) {
    Remove-Item -LiteralPath $stagingPath -Recurse -Force
}
New-Item -ItemType Directory -Path (Join-Path $stagingPath "backend") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stagingPath "deployment") -Force | Out-Null

try {
    Copy-Item (Join-Path $projectRoot "asgi.py") $stagingPath
    Copy-Item (Join-Path $projectRoot "requirements.txt") $stagingPath
    Copy-Item (Join-Path $projectRoot "backend\*.py") (Join-Path $stagingPath "backend")
    Copy-Item (Join-Path $projectRoot "backend\migrations") (Join-Path $stagingPath "backend") -Recurse
    Copy-Item (Join-Path $projectRoot "model") $stagingPath -Recurse
    Copy-Item (Join-Path $projectRoot "out") $stagingPath -Recurse
    Copy-Item (Join-Path $PSScriptRoot "pythonanywhere.env.example") (Join-Path $stagingPath "deployment")
    Compress-Archive -Path (Join-Path $stagingPath "*") -DestinationPath $packagePath -Force
}
finally {
    Remove-Item -LiteralPath $stagingPath -Recurse -Force
}

Write-Output "Created $packagePath"
