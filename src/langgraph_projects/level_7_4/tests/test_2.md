$body = @{
thread_id = "1"
user_id = "Matthew"
todo_kind = "personal"
message = "My wife asked me to book dance lessons for daughter."
} | ConvertTo-Json -Depth 10

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat"  -ContentType "application/json" -Body $body
$resp.reply
