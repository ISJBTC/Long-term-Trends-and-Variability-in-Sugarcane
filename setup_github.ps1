# Run this script from inside the "Long-term Trends and Variability in Sugarcane" folder
# Prerequisites: git, GitHub CLI (gh) installed and authenticated (gh auth login)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Initializing git..."
git init

Write-Host "Adding files..."
git add -A

Write-Host "Creating initial commit..."
git commit -m "Initial commit: reproducibility package for five-district sugarcane analysis"

Write-Host "Creating GitHub repository and pushing..."
gh repo create "Long-term-Trends-and-Variability-in-Sugarcane" `
  --public `
  --source=. `
  --remote=origin `
  --push `
  --description "Reproducibility package for five-district sugarcane production analysis (Maharashtra and Karnataka, India)"

Write-Host "Done. Repository URL:"
git remote get-url origin
