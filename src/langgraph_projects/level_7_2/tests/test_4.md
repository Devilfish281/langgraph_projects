$body = @{
thread_id = "1"
user_id = "Matthew"
message = "I need to fix the jammed electric Yale lock on the door."
} | ConvertTo-Json

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `  -ContentType "application/json"` -Body $body

$resp.reply
