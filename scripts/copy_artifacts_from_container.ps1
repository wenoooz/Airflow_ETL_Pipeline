param(
    [Parameter(Mandatory=$true)]
    [string]$RunId,
    [string]$ContainerName = "airflow-scheduler"
)
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$Dest = Join-Path $ProjectRoot "artifacts" $RunId
$Src = "/opt/airflow/artifacts/$RunId"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
docker cp "${ContainerName}:${Src}/." $Dest
Write-Host "Copied to $Dest"
