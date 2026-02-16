$resp = Invoke-RestMethod "http://localhost:8000/instructions/Matthew"
$resp | ConvertTo-Json -Depth 10
