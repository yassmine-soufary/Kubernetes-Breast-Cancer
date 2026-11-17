# Teste l'API à travers l'Ingress (ou un port-forward) - PowerShell
#   .\scripts\test-api.ps1                              -> http://bc.local
#   .\scripts\test-api.ps1 -Base http://localhost:8000  (après kubectl port-forward svc/bc-api 8000:8000)
param(
    [string]$Base = "http://bc.local",
    [string]$Key = "k8s-tp-key"
)
$headers = @{ "X-API-Key" = $Key }

Write-Host "== info"
Invoke-RestMethod "$Base/api/info" | Select-Object app_version, model_version, algo, pod | Format-List

Write-Host "== classification (tumeur plutôt maligne)"
$malin = @{ features = @{ "mean radius" = 20.5; "mean texture" = 21.0; "mean perimeter" = 135.0;
                          "mean area" = 1300.0; "worst radius" = 25.0; "worst concave points" = 0.2 } } |
         ConvertTo-Json
Invoke-RestMethod -Method Post "$Base/api/predict/classification" -Headers $headers `
    -ContentType "application/json" -Body $malin | Format-List

Write-Host "== régression (mean area)"
$reg = @{ features = @{ "mean radius" = 13.5; "mean perimeter" = 87.0 } } | ConvertTo-Json
Invoke-RestMethod -Method Post "$Base/api/predict/regression" -Headers $headers `
    -ContentType "application/json" -Body $reg | Format-List

Write-Host "== stats (répéter : le pod qui répond change)"
1..5 | ForEach-Object { Invoke-RestMethod "$Base/api/stats" | Select-Object served_by, classification, app_version }
