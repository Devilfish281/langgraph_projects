$body = @{
  thread_id = "2"
  user_id   = "Matthew"
  message   = "Yes, give me some options to call for swim lessons."
} | ConvertTo-Json -Compress

$body | curl.exe --% -N -X POST "http://localhost:8000/stream" -H "Content-Type: application/json" --data-binary "@-"
