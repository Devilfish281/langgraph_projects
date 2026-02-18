Invoke-RestMethod -Method Get -Uri "http://localhost:8000/todos/work/Matthew" |
ConvertTo-Json -Depth 20
