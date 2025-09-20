from config import CONSUL_HOST, CONSUL_PORT, SERVICE_NAME, SEARXNG_URL, SHARED_DIR
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from consul_utils import register_service, deregister_service
from enum import Enum
import httpx
from typing import Optional, Dict, Any, List
import consul
import time
import subprocess
import shutil
import socket
import glob
import uuid
import os
import atexit
import requests
import traceback
import json

# 例子，后续需要将其中的模型名字进行规范
class ModelName(str, Enum):
    silicon_flow = "silicon-flow"
    moonshot = "moonshot"
    deepseek = "deepseek"
    Qwen = "Qwen"

MODEL_TO_APIKEY = {
    "silicon-flow": "app-OFaVMpobX30c0Tv36i1luC2U", 
    "moonshot": "app-p9c2JEIrsJariPYeIxU3otjB",
    "deepseek": "app-zJT765lCyC0UkeNqRik4vYRw",
    "Qwen": "app-VH46JNigYuWdqf62sBucCOcw"
}

dify_url = "http://10.92.64.224/v1/chat-messages"

class ChatRequest(BaseModel):
    model: ModelName
    question: str
    requestId: str
    use_web_search: Optional[bool]
    user_id: Optional[str] # 默认为匿名用户
    conversation_id: Optional[str] = None # 默认为新对话

class ChatResponse(BaseModel):
    answer: str
    requestId: str
    conversation_id: Optional[str] = None # 默认为新对话

# 数据处理请求模型 - 支持动态数据字段
class DataProcessRequest(BaseModel):
    model: ModelName  # 模型名称
    user_prompt: str  # 用户需求描述
    user_id: Optional[str] = "defaultid"  # 用户ID
    
    # 动态字段将通过 __init__ 处理
    def __init__(self, **data):
        # 提取所有以 'data' 开头的字段
        self.data_fields = {}
        regular_fields = {}
        
        for key, value in data.items():
            if key.startswith('data') and key[4:].isdigit():
                self.data_fields[key] = value
            else:
                regular_fields[key] = value
        
        super().__init__(**regular_fields)

# 数据处理响应模型
class DataProcessResponse(BaseModel):
    status: str  # success 或 error
    result: Optional[List[Dict[str, Any]]] = None  # 处理后的数据结果
    answer: Optional[str] = None  # 大模型的回答
    error_details: Optional[str] = None

app = FastAPI()

# 添加服务启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """服务启动时注册到Consul"""
    service_id = start_call_llm_service()
    if service_id:
        app.state.service_id = service_id
        print(f"call_llm服务已注册到Consul，服务ID: {service_id}")

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时从Consul注销"""
    if hasattr(app.state, 'service_id'):
        deregister_service(app.state.service_id)

def start_call_llm_service():
    SERVICE_PORT = 8000
    tags = ['llm', 'ai', 'dify']
    service_id = register_service(SERVICE_PORT, tags)
    return service_id

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": SERVICE_NAME}

def perform_web_search(query: str) -> str:
    try:
        search_url = f"{SEARXNG_URL}/search"
        print(f"正在搜索: {query}")
        print(f"搜索URL: {search_url}")

        resp = requests.get(
            search_url,
            params={"q": query, "format": "json"},
            timeout=10
        )

        print(f"搜索响应状态码: {resp.status_code}")

        if resp.status_code != 200:
            return f"【联网搜索失败】：HTTP {resp.status_code} - {resp.text}\n"

        try:
            json_data = resp.json()
        except ValueError as e:
            return f"【联网搜索失败】：无法解析JSON响应 - {e}\n"

        results = json_data.get("results", [])
        if not results:
            return f"【联网搜索结果】：未找到相关信息\n"

        top_results = results[:3]
        formatted = "\n".join([
            f"{i + 1}. {r.get('title', '无标题')}\nURL: {r.get('url', '无URL')}\n摘要: {r.get('content', '无摘要')}"
            for i, r in enumerate(top_results)
        ])

        return f"【以下为联网搜索结果】：\n{formatted}\n"

    except requests.exceptions.ConnectionError as e:
        print(f"连接错误: {e}")
        return f"【联网搜索失败】：无法连接到搜索服务 ({SEARXNG_URL})\n"
    except requests.exceptions.Timeout as e:
        print(f"超时错误: {e}")
        return f"【联网搜索失败】：搜索服务响应超时\n"
    except Exception as e:
        print(f"未知错误: {e}")
        return f"【联网搜索失败】：{e}\n"

async def call_dify(model: str, prompt: str, user_id: str, conversation_id: Optional[str] = None) -> tuple:
    api_key = MODEL_TO_APIKEY.get(model)

    if not api_key:
        return f"[error]模型{model}未配置API KEY", None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": {},
        "query": prompt,
        "response_mode": "blocking",
        "user": user_id,
        "conversation_id": conversation_id
    }

    timeout = httpx.Timeout(120.0, read=120.0, connect=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(dify_url, headers=headers, json=data)

            print("状态码:", resp.status_code)
            print("原始内容:", resp.text)

            if resp.status_code == 504:
                raise HTTPException(status_code=504, detail="[Dify错误]模型响应超时，稍后再试")

            try:
                result = resp.json()
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"[响应格式错误]无法解析JSON:{e}\n原始响应:{resp.text}")

            if "answer" in result:
                return result["answer"], result.get("conversation_id")
            elif "message" in result:
                raise HTTPException(status_code=502, detail=f"[Dify错误] {result['message']}")
            else:
                raise HTTPException(status_code=502, detail="[Dify响应格式异常]")

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="[超时] Dify 响应超时")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"[请求失败] {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"[未知错误] {e}")

# 聊天接口
@app.post("/llm", response_model=ChatResponse)
async def get_model(request: ChatRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    question = request.question.strip()

    if request.use_web_search:
        try:
            web_result = perform_web_search(question).strip()
            if not web_result:
                raise ValueError("Empty web result")

            full_prompt = (
                f"你是一名知识渊博的智能助手。\n\n"
                f"以下是与用户问题相关的最新搜索信息：\n"
                f"{web_result}\n\n"
                f"请根据以上资料，结合用户的问题，进行精准和详尽的解答。\n\n"
                f"【用户问题】：{question}"
            )
        except Exception as e:
            full_prompt = (
                "【提示】：联网搜索失败，以下为基于已有知识的回答。\n\n"
                f"【用户问题】：{question}"
            )
    else:
        full_prompt = question

    answer, new_conversation_id = await call_dify(
        request.model,
        full_prompt,
        user_id=request.user_id or "defaultid",
        conversation_id=str(request.conversation_id) if request.conversation_id else None
    )

    return ChatResponse(
        answer=answer,
        requestId=request.requestId,
        conversation_id=new_conversation_id
    )

# 数据处理专用的Dify调用函数
async def call_dify_with_tools(model: str, prompt: str, data_dict: Dict[str, List[Dict]], 
                               user_id: Optional[str] = "defaultid", 
                               conversation_id: Optional[str] = None) -> Dict[str, Any]:
    """
    调用配置了工具函数的Dify应用
    Dify会根据用户需求自动选择合适的工具函数并调用
    """
    api_key = MODEL_TO_APIKEY.get(model)

    if not api_key:
        raise HTTPException(status_code=400, detail=f"[error]模型{model}未配置API KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 将数据转换为Dify工具函数期望的file_content格式
    # file_content应该是List[str]，每个元素是JSON字符串
    file_content = []
    for key, value in data_dict.items():
        # 将每个数据集转换为JSON字符串
        json_str = json.dumps(value, ensure_ascii=False)
        file_content.append(json_str)

    # 构建inputs，包含file_content和其他必要参数
    data_inputs = {
        "file_content": file_content,  # 这是Dify工具函数期望的格式
        "user_prompt": prompt  # 用户的原始需求
    }

    data = {
        "inputs": data_inputs,
        "query": prompt,  # 用户的处理需求描述
        "response_mode": "blocking",
        "user": user_id,
        "conversation_id": conversation_id
    }

    print("=== 调试信息 ===")
    print("发送到Dify的数据:")
    print(f"- file_content包含 {len(file_content)} 个数据集")
    for i, content in enumerate(file_content):
        data_length = len(json.loads(content)) if content else 0
        print(f"  - 数据集 {i}: {data_length} 条记录")
    print("- query:", prompt)
    print("===============")

    timeout = httpx.Timeout(120.0, read=120.0, connect=10.0)

    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            resp = await client.post(dify_url, headers=headers, json=data)

            print("状态码:", resp.status_code)
            print("原始内容:", resp.text)

            if resp.status_code == 504:
                raise HTTPException(status_code=504, detail="[Dify错误]模型响应超时，稍后再试")

            try:
                result = resp.json()
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"[响应格式错误]无法解析JSON:{e}\n原始响应:{resp.text}")

            if "answer" in result:
                return {
                    "answer": result["answer"],
                    "conversation_id": result.get("conversation_id")
                }
            elif "message" in result:
                raise HTTPException(status_code=502, detail=f"[Dify错误] {result['message']}")
            else:
                raise HTTPException(status_code=502, detail="[Dify响应格式异常]")

    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail="[超时] Dify 响应超时")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"[请求失败] {e}")
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"[未知错误] {repr(e)}\n{tb}")

# 统一的数据处理接口
@app.post("/data-process/execute", response_model=DataProcessResponse)
async def execute_data_process(request: Dict[str, Any]) -> DataProcessResponse:
    """
    统一的数据处理接口
    大模型会在Dify中自动选择合适的工具函数处理数据
    
    请求格式:
    {
        "model": "model_name",
        "user_prompt": "用户需求描述",
        "user_id": "用户ID",
        "data0": [{"col1": "value1", "col2": "value2"}, ...],  # 第一个数据集
        "data1": [{"col1": "value1", "col2": "value2"}, ...],  # 第二个数据集（可选）
        ...
    }
    """
    try:
        # 提取基本参数
        model = request.get("model")
        user_prompt = request.get("user_prompt")
        user_id = request.get("user_id", "defaultid")
        
        if not model or not user_prompt:
            return DataProcessResponse(
                status="error",
                error_details="缺少必要参数: model 和 user_prompt"
            )

        # 提取数据字段 (data0, data1, data2, ...)
        data_dict = {}
        for key, value in request.items():
            if key.startswith('data') and key[4:].isdigit():
                if not isinstance(value, list):
                    return DataProcessResponse(
                        status="error",
                        error_details=f"数据字段 {key} 必须是列表格式"
                    )
                data_dict[key] = value

        if not data_dict:
            return DataProcessResponse(
                status="error",
                error_details="未找到任何数据字段 (data0, data1, ...)"
            )

        print(f"接收到数据处理请求: {len(data_dict)} 个数据集")
        for key, value in data_dict.items():
            print(f"- {key}: {len(value)} 条记录")

        # 调用Dify，让大模型决策并调用工具函数
        result = await call_dify_with_tools(
            model=model,
            prompt=user_prompt,
            data_dict=data_dict,
            user_id=user_id
        )

        # Dify的工具函数会自动处理数据并返回结果
        # 大模型会根据工具函数的返回结果生成最终回答
        return DataProcessResponse(
            status="success",
            answer=result.get("answer"),
            # 注意: 实际的处理结果数据需要从answer中解析
            # 或者根据具体的Dify工具函数返回格式来调整
        )

    except Exception as e:
        tb = traceback.format_exc()
        return DataProcessResponse(
            status="error",
            error_details=f"{repr(e)}\n{tb}"
        )

if __name__ == "__main__":
    import uvicorn

    # 注册服务到Consul
    service_id = register_service()

    # 程序退出时注销服务
    if service_id:
        atexit.register(deregister_service, service_id)

    SERVICE_PORT = 8000
    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
