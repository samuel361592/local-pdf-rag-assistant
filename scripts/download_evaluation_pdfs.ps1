[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$targetDirectory = Join-Path $projectRoot "documents\evaluation"

$documents = @(
    [pscustomobject]@{
        Name = "NIST.AI.100-1.pdf"
        Url = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
        ExpectedSha256 = "7576EDB531D9848825814EE88E28B1795D3A84B435B4B797D3670EAFDC4A89F1"
    },
    [pscustomobject]@{
        Name = "NIST.AI.600-1.pdf"
        Url = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf"
        ExpectedSha256 = "6E73620AB6B64E90EF2C04BF0E0D6246185A2F4B1B13CAB0DF494496CFF89B6A"
    }
)

if (-not (Test-Path -LiteralPath $targetDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $targetDirectory | Out-Null
}

foreach ($document in $documents) {
    $destination = Join-Path $targetDirectory $document.Name

    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        $existingHash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash
        if ($existingHash -eq $document.ExpectedSha256) {
            Write-Host "Verified existing file: $($document.Name)"
            continue
        }
        Write-Host "Existing file hash differs; downloading a verified copy: $($document.Name)"
    }
    else {
        Write-Host "Downloading: $($document.Name)"
    }

    $temporaryName = ".{0}.{1}.download" -f $document.Name, [guid]::NewGuid().ToString("N")
    $temporaryPath = Join-Path $targetDirectory $temporaryName

    try {
        Invoke-WebRequest `
            -Uri $document.Url `
            -OutFile $temporaryPath `
            -UseBasicParsing

        $downloadedHash = (
            Get-FileHash -LiteralPath $temporaryPath -Algorithm SHA256
        ).Hash
        if ($downloadedHash -ne $document.ExpectedSha256) {
            throw (
                "SHA-256 verification failed for {0}. Expected {1}, received {2}." -f `
                    $document.Name,
                    $document.ExpectedSha256,
                    $downloadedHash
            )
        }

        Move-Item -LiteralPath $temporaryPath -Destination $destination -Force
        Write-Host "Downloaded and verified: $($document.Name)"
    }
    finally {
        if (Test-Path -LiteralPath $temporaryPath -PathType Leaf) {
            Remove-Item -LiteralPath $temporaryPath -Force
        }
    }
}

Write-Host "Evaluation PDFs are ready in: $targetDirectory"
