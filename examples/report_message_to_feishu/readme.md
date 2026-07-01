前置条件: 获取app_id, app_secret, open_id/chat_id。

app_id, app_secret获取方法:
在飞书上创建一个机器人(通过网站https://open.feishu.cn/app), 并拿到app_id, app_secret。

open_id获取方法: !!!!!如果让机器人主动上报相关信息,需要用到open_id/chat_id。如果是agent聊天不需要用到open_id/chat_id。
然后在网站https://open.feishu.cn/document/server-docs/im-v1/message/create中的示例教程中获取open_id(一个飞书用户的唯一标识)或chat_id(一个飞书群的唯一标识)。
进入调试页面:
1. 先指定应用(也就是刚才创建的机器人应用)
2. 然后找到receive_id_type选择open_id/chat_id
3. 然后点击快速复制open_id/chat_id,选择对应用户就能获取到open_id/chat_id

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

事件开通需要的权限列表:
im.message.receive_v1: 接收飞书消息