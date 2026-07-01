前置条件: 获取app_id, app_secret, open_id/chat_id。

app_id, app_secret获取方法:
在飞书上创建一个机器人(通过网站https://open.feishu.cn/app), 并拿到app_id, app_secret。

注: 向同一用户发送消息的限频为 5 QPS


机器人权限开通需要的权限列表:
{
  "scopes": {
    "tenant": [
      "im:message.reactions:write_only",
      "contact:user.employee_id:readonly",
      "event:ip_list",
      "im:chat:readonly",
      "im:message",
      "im:message.group_at_msg:readonly",
      "im:message.group_msg",
      "im:message.p2p_msg:readonly",
      "im:message:send_as_bot",
      "im:resource",
      "speech_to_text:speech"
    ],
    "user": []
  }
}

事件开通需要的权限列表: 注: 此功能必须先跑一次链接才可以开通
im.message.receive_v1: 接收飞书消息