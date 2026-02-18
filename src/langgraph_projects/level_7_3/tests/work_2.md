$body = @{
thread_id = "1"
user_id = "Matthew"
todo_kind = "work"
message = "Work: submit expense report by Friday. Takes 20 minutes."
} | ConvertTo-Json -Depth 10

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" -ContentType "application/json" -Body $body
$resp | ConvertTo-Json -Depth 10
