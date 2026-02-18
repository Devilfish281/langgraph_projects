$body = @{
thread_id = "1"
user_id = "Matthew"
todo_kind = "work"
message = "Add a work task: finalize the Q1 report. It should take 60 minutes and is due Friday."
} | ConvertTo-Json -Depth 5

$params = @{
Method = "Post"
Uri = "http://localhost:8000/chat"
ContentType = "application/json"
Body = $body
}

$resp = Invoke-RestMethod @params
$resp.reply
