Get-Process -Name HawaqmAI.Api -ErrorAction SilentlyContinue | Stop-Process -Force
cd "D:\DOH Project 2026\DOHWorkspace\AQMSAIChatService\src\HawaqmAI.Api"
dotnet run --launch-profile Development



Database insertion
cd "d:\DOH Project 2026\DOHWorkspace\DatabaseKnowledge"
python live_data_with_backfill.py
