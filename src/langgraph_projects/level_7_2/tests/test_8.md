$body = @{
thread_id = "2"
message = "Yes, give me some options to call for swim lessons."
} | ConvertTo-Json

$resp = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/chat" `  -ContentType "application/json"` -Body $body
$resp.reply
