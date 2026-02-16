$body = @{
thread_id = "1"
user_id = "Matthew"
message = "My name is Matthew. I live in SF with my wife. I have a 1 year old daughter."
} | ConvertTo-Json

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `  -ContentType "application/json"`
-Body $body

$resp.reply
