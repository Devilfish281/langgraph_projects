$resp = Invoke-RestMethod "http://localhost:8000/todos/personal/Matthew"
$resp.todos | ConvertTo-Json -Depth 20
