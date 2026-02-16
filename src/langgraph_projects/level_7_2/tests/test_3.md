$body = @{
thread_id = "1"
user_id = "Matthew"
message = "When creating or updating ToDo items, include specific local businesses / vendors."
} | ConvertTo-Json

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `  -ContentType "application/json"`
-Body $body

$resp.reply
