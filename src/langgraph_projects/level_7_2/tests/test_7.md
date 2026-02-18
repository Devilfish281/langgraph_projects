$body = @{
thread_id = "2"
user_id = "Matthew"
message = "I have 30 minutes, what tasks can I get done?"
} | ConvertTo-Json

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `  -ContentType "application/json"` -Body $body
$resp.reply
