param(
    [string]$inputFile = "audio.wav",
    [string]$outputDir = "/output",
    [string]$model = "large-v3",
    [string]$language = "ja"
)

# Change to the project root directory where docker-compose.yml is located
$projectRoot = (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))))
Set-Location -Path $projectRoot

# Check if docker-compose.yml exists
$dockerComposePath = Join-Path $projectRoot "docker-compose.yml"
if (-not (Test-Path $dockerComposePath)) {
    Write-Error "Docker Compose file not found: $dockerComposePath"
    exit 1
}

# Check if input file exists
$inputPath = Join-Path $projectRoot "input" $inputFile
if (-not (Test-Path $inputPath)) {
    Write-Error "Input file not found: $inputPath"
    exit 1
}

# Create output directory if it doesn't exist
$outputPath = Join-Path $projectRoot "output"
if (-not (Test-Path $outputPath)) {
    New-Item -ItemType Directory -Path $outputPath
}

Write-Host "Using Docker Compose file: $dockerComposePath"
Write-Host "Processing file: $inputFile"
Write-Host "Input path: $inputPath"
Write-Host "Output path: $outputPath"

docker compose run --rm whisper `
    --model $model `
    --language $language `
    "/input/$inputFile" `
    --output_dir $outputDir `
    --output_format json