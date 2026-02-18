Invoke-RestMethod -Method Get -Uri "http://localhost:8000/todos/personal/Matthew" |
ConvertTo-Json -Depth 20
