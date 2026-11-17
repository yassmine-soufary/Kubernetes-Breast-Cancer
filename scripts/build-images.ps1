# Construit les 4 images du TP puis les charge dans Minikube (imagePullPolicy: IfNotPresent)
# Usage (PowerShell, depuis la racine du projet) : .\scripts\build-images.ps1
$ErrorActionPreference = "Stop"

docker build -t bc-trainer:1.0 app/trainer
docker build -t bc-api:1.0 --build-arg APP_VERSION=1.0 app/api
docker build -t bc-api:2.0 --build-arg APP_VERSION=2.0 app/api
docker build -t bc-front:1.0 app/front

foreach ($img in @("bc-trainer:1.0", "bc-api:1.0", "bc-api:2.0", "bc-front:1.0")) {
    Write-Host ">> minikube image load $img"
    minikube image load $img
}

minikube image ls | Select-String "bc-"
