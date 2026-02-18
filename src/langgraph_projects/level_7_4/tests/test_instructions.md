$resp = Invoke-RestMethod "http://localhost:8000/instructions/personal/Matthew"
$resp | ConvertTo-Json -Depth 10
