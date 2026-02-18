$body = @{
thread_id = "1"
user_id = "Matthew"
todo_kind = "personal"
message = "Need to take the Lexus car in for 100,000 miles service."
} | ConvertTo-Json -Depth 10

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `  -ContentType "application/json"` -Body $body
$resp.reply
