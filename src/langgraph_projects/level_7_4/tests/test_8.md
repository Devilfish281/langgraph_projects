$body = @{
thread_id = "2"
user_id = "Matthew"
todo_kind = "personal"
message = "Yes, give me some options to call for dance lessons."
} | ConvertTo-Json -Depth 10

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `  -ContentType "application/json"` -Body $body
$resp.reply
