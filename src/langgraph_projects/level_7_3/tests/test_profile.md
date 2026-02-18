$resp = Invoke-RestMethod "http://localhost:8000/profile/Matthew"
$resp | ConvertTo-Json -Depth 10
