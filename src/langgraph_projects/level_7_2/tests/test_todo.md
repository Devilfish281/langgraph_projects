$resp = Invoke-RestMethod "http://localhost:8000/todos/Matthew"
$resp.todos | ConvertTo-Json -Depth 20
