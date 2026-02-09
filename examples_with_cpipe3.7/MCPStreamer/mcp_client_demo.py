import time
import requests
import json


if __name__ == "__main__":
    MCP_URL = "http://localhost:19966/mcp"  # 替换为实际 MCP 服务地址
    

    # step1: initialize
    payload = {"method":"initialize","params":{
        "protocolVersion":"2025-03-26","capabilities":{},
        "clientInfo":{"name":"Cherry Studio","version":"1.4.7"}},
        "jsonrpc":"2.0","id":0}
    
    response = requests.post(
            url=MCP_URL,
            headers={"Content-Type": "application/json",
                     "accept":"application/json, text/event-stream"
                     },
            data=json.dumps(payload)
        )
    session_id = response.headers['mcp-session-id']
    print(response.text)

    #step2: initialized
    payload = {"method":"notifications/initialized","jsonrpc":"2.0"} #2
    
    response = requests.post( 
            url=MCP_URL,
            headers={"Content-Type": "application/json",
                     "accept":"application/json, text/event-stream",
                     "mcp-session-id":session_id
                     },
            data=json.dumps(payload)
        )
    print(response.text)

    # step3: get tools list (option)
    print("**************tools/list**************") # 可掉可不掉
    payload = {"method":"tools/list",
            "params":{},
            "jsonrpc":"2.0","id":1}
    
    response = requests.post(
            url=MCP_URL,
            headers={"Content-Type": "application/json",
                     "accept":"application/json, text/event-stream",
                     "mcp-session-id":session_id
                     },
            data=json.dumps(payload)
        )
    print(response.text)

    # step4: tools/call
    # img_base641 = ""
    # with open("mcp_/1752049034722.txt", "r") as image_file:
    #     img_base641 = image_file.read()
    # img_base642 = ""
    # with open("mcp_/1111.txt", "r") as image_file:
    #     img_base642 = image_file.read()
    i = 0
    while True:
    # third tools/call
        print("**************tools/call**************")
        # data = img_base641 if i % 2 == 0 else img_base642
        payload = {"method":"tools/call",
                "params":{"name":"inference","arguments":
                {
                # "image_url":"http://code-x.oss-cn-hangzhou.aliyuncs.com/zh/1752049034722.jpg",
                # "image_base64":data,
                "image_path":"examples/create_MCPStreamer/2.png",
                "result_inference_image": False
                }
                },
            "jsonrpc":"2.0","id":2}
        
        response = requests.post(
                url=MCP_URL,
                headers={"Content-Type": "application/json",
                        "accept":"application/json, text/event-stream",
                        "mcp-session-id":session_id
                        },
                data=json.dumps(payload)
            )
        print(response.text)
        time.sleep(1)