$body = @{
thread_id = "1"
user_id = "Matthew"
message = "My wife asked me to book dance lessons for daughter."
} | ConvertTo-Json

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `  -ContentType "application/json"` -Body $body
$resp.reply
