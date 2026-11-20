工作流: `VideoStreamer -> YOLO26 -> Tracker -> my_logic -> AlarmReport -> CPipeInsight`

`my_logic` 只负责判断何时报警,通过 `new_cdata.add_alarm(stream_name, Alarm(...))` 把报警交给 `AlarmReport`,截图和录像由 `AlarmReport` 完成:

| 场景(X/Y 帧投票确认) | 报警类型 | 上游提供 | 效果 |
| --- | --- | --- | --- |
| 区域从无人变为有人 | `video` | `alarm_text` + 出现人员的 `track_ids` | 当前帧截图 + 前后 3 秒视频,出现的人红框 |
| 区域从有人变为无人 | `video` | `alarm_text` + 刚离开人员的 `track_ids` | 走光那一帧截图 + 前后 3 秒视频,视频中离开的人红框 |

`video` 类型报警会同时生成截图(当前帧)和视频,前后时长由 `AlarmReport(video_seconds=3)` 配置。

报警数据保存在 `./alarm_data`:

- `alarm.db`: SQLite 数据库,`alarms` 表为报警记录,`media_files` / `alarm_media` 为图片和视频文件
- `YYYY-MM-DD/*.jpg`、`YYYY-MM-DD/clip_*.mp4`: 报警截图与视频(有 ffmpeg 时为浏览器可直接播放的 H.264)

查看最近的报警:

```bash
sqlite3 alarm_data/alarm.db "select alarm_time_str, alarm_name, alarm_text, alarm_type, image_status, video_status from alarms order by alarm_time desc limit 10"
```

配置 `AlarmReport(mqtt_broker="192.168.10.7")` 后,每条报警会实时发布到 `cpipe/alarm/{流名称}`。