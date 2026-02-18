$body = @{
thread_id = "1"
user_id = "Matthew"
todo_kind = "personal"
message = "My name is Matthew. I live in San Jose with my wife. I have a 5 year old daughter."
} | ConvertTo-Json -Depth 5

$params = @{
Method = "Post"
Uri = "http://localhost:8000/chat"
ContentType = "application/json"
Body = $body
}

$resp = Invoke-RestMethod @params
$resp.reply
