param([Parameter(Mandatory=$true)][string]$InputPath, [Parameter(Mandatory=$true)][string]$OutputPath)
$ErrorActionPreference = 'Stop'
$wordApp = $null
$wordDoc = $null
try {
    $wordApp = New-Object -ComObject Word.Application
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $confirmConversion = $false
    $readOnly = $true
    $addToRecent = $false
    $wordDoc = $wordApp.Documents.Open([ref]$InputPath, [ref]$confirmConversion, [ref]$readOnly, [ref]$addToRecent)
    $wordDoc.Repaginate()
    $wordDoc.ExportAsFixedFormat($OutputPath, 17)
} finally {
    $saveChanges = 0
    if ($null -ne $wordDoc) { $wordDoc.Close([ref]$saveChanges); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordDoc) }
    if ($null -ne $wordApp) { $wordApp.Quit([ref]$saveChanges); [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($wordApp) }
}
