$body = @{
thread_id = "2"
user_id = "Matthew"
message = "Yes, give me some options to call for swim lessons."
} | ConvertTo-Json -Compress

$req = [System.Net.HttpWebRequest]::Create("http://localhost:8000/stream")
$req.Method = "POST"
$req.ContentType = "application/json"
$req.Accept = "text/event-stream"

$bytes = [System.Text.Encoding]::UTF8.GetBytes($body)
$req.ContentLength = $bytes.Length
$stream = $req.GetRequestStream()
$stream.Write($bytes,0,$bytes.Length)
$stream.Close()

$resp = $req.GetResponse()
$reader = New-Object System.IO.StreamReader($resp.GetResponseStream())

while (-not $reader.EndOfStream) {
  $line = $reader.ReadLine()
  if ($line) { $line }
}
