$body = @{
thread_id = "1"
user_id = "Matthew"
todo_kind = "personal"
message = "For the Dance lessons, I need to get that done by end of November."
} | ConvertTo-Json -Depth 10

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `  -ContentType "application/json"` -Body $body
$resp.reply
