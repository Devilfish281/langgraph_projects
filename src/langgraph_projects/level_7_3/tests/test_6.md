$body = @{
thread_id = "1"
user_id = "Matthew"
todo_kind = "personal"
message = "Need to call back City Toyota to schedule car service."
} | ConvertTo-Json -Depth 10

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `  -ContentType "application/json"` -Body $body
$resp.reply
