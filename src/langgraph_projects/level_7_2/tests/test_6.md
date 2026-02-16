$body = @{
thread_id = "1"
user_id = "Matthew"
message = "Need to call back City Toyota to schedule car service."
} | ConvertTo-Json

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `  -ContentType "application/json"`
-Body $body

$resp.reply
