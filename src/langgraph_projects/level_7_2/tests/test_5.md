$body = @{
thread_id = "1"
user_id = "Matthew"
todo_kind = "personal"
message = "For the dance lessons, I need to get that done by end of November."
} | ConvertTo-Json

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `  -ContentType "application/json"` -Body $body

$resp.reply
