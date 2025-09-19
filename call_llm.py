from config import CONSUL_HOST, CONSUL_PORT, SERVICE_NAME, SEARXNG_URL, SHARED_DIR
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from consul_utils import register_service, deregister_service
from enum import Enum
import httpx
from typing import Optional, Dict, Any, List
from hdfs import InsecureClient
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
import re

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


# 数据处理请求模型
class NewDataProcessRequest(BaseModel):
    model: ModelName  # 模型名称
    user_prompt: str  # 用户需求描述
    # 动态数据字段，支持 data0, data1, data2 等
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
class NewDataProcessResponse(BaseModel):
    status: str  # success 或 error
    result: Optional[List[Dict[str, Any]]] = None  # 处理后的数据结果
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
    # 这里可以添加更多服务启动后的逻辑，比如启动FastAPI应用等
    return service_id


# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": SERVICE_NAME}


def perform_web_search(query: str) -> str:
    try:
        # 使用正确的变量
        search_url = f"{SEARXNG_URL}/search"
        print(f"正在搜索: {query}")
        print(f"搜索URL: {search_url}")

        resp = requests.get(
            search_url,
            params={"q": query, "format": "json"},
            timeout=10  # 适当增加超时时间
        )

        print(f"搜索响应状态码: {resp.status_code}")

        # 检查HTTP状态码
        if resp.status_code != 200:
            return f"【联网搜索失败】：HTTP {resp.status_code} - {resp.text}\n"

        try:
            json_data = resp.json()
        except ValueError as e:
            return f"【联网搜索失败】：无法解析JSON响应 - {e}\n"

        results = json_data.get("results", [])
        if not results:
            return f"【联网搜索结果】：未找到相关信息\n"

        # 取前3个结果
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

async def call_dify(model: str, prompt: str, user_id: str, conversation_id: Optional[str] = None) -> str:
    api_key = MODEL_TO_APIKEY.get(model)

    if not api_key:
        return f"[error]模型{model}未配置API KEY"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": {},
        "query": prompt,
        "response_mode": "blocking",
        "user": user_id,
        "conversation_id": conversation_id   # 若有上下文则填
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
                result = resp.json()  # 只在成功时赋值
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


# 接口的返回值应当符合ChatResponse的Pydantic模型结构
@app.post("/llm", response_model=ChatResponse)
async def get_model(request: ChatRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    # 原始问题
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
            # 联网失败，降级处理
            full_prompt = (
                "【提示】：联网搜索失败，以下为基于已有知识的回答。\n\n"
                f"【用户问题】：{question}"
            )
    else:
        # 不使用联网搜索时直接使用原问题
        full_prompt = question

    # 调用 Dify 接口
    answer, new_conversation_id = await call_dify(
        request.model,
        full_prompt,
        user_id=request.user_id,
        conversation_id=str(request.conversation_id) if request.conversation_id else None
    )

    return ChatResponse(
        answer=answer,
        requestId=request.requestId,
        conversation_id=new_conversation_id
    )

# 数据处理部分调用大模型
import re

def extract_json_from_text(text: str):
    """从文本中提取JSON内容"""
    try:
        # 方法1: 查找被```json```包围的代码块
        json_pattern = r'```json\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```'
        matches = re.findall(json_pattern, text, re.MULTILINE | re.DOTALL)
        if matches:
            return json.loads(matches[0])
        
        # 方法2: 查找被```包围的代码块（不一定有json标记）
        code_pattern = r'```\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```'
        matches = re.findall(code_pattern, text, re.MULTILINE | re.DOTALL)
        if matches:
            return json.loads(matches[0])
        
        # 方法3: 直接查找JSON数组或对象
        # 查找数组
        array_pattern = r'\[(?:[^[\]]*|(?:\[(?:[^[\]]*|(?:\[[^\]]*\])*)*\]))*\]'
        array_matches = re.findall(array_pattern, text)
        for match in array_matches:
            try:
                parsed = json.loads(match)
                # 验证是否是我们想要的数据结构
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # 查找对象
        object_pattern = r'\{(?:[^{}]*|(?:\{(?:[^{}]*|(?:\{[^}]*\})*)*\}))*\}'
        object_matches = re.findall(object_pattern, text)
        for match in object_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except json.JSONDecodeError:
                continue
                
        return None
        
    except Exception as e:
        print(f"JSON提取失败: {e}")
        return None

def extract_json_from_text(text: str):
    """从文本中提取JSON内容"""
    try:
        # 方法1: 查找被```json```包围的代码块
        json_pattern = r'```json\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```'
        matches = re.findall(json_pattern, text, re.MULTILINE | re.DOTALL)
        if matches:
            return json.loads(matches[0])
        
        # 方法2: 查找被```包围的代码块（不一定有json标记）
        code_pattern = r'```\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```'
        matches = re.findall(code_pattern, text, re.MULTILINE | re.DOTALL)
        if matches:
            return json.loads(matches[0])
        
        # 方法3: 直接查找JSON数组或对象
        # 查找数组
        array_pattern = r'\[(?:[^[\]]*|(?:\[(?:[^[\]]*|(?:\[[^\]]*\])*)*\]))*\]'
        array_matches = re.findall(array_pattern, text)
        for match in array_matches:
            try:
                parsed = json.loads(match)
                # 验证是否是我们想要的数据结构
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed
            except json.JSONDecodeError:
                continue
        
        # 查找对象
        object_pattern = r'\{(?:[^{}]*|(?:\{(?:[^{}]*|(?:\{[^}]*\})*)*\}))*\}'
        object_matches = re.findall(object_pattern, text)
        for match in object_matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except json.JSONDecodeError:
                continue
                
        return None
        
    except Exception as e:
        print(f"JSON提取失败: {e}")
        return None


async def call_dify_data_tool(model: str, prompt: str, data_dict: Dict[str, List[Dict]], 
                              user_id: Optional[str] = "defaultid", 
                              conversation_id: Optional[str] = None) -> Dict[str, Any]:
    api_key = MODEL_TO_APIKEY.get(model)

    if not api_key:
        raise HTTPException(status_code=400, detail=f"[error]模型{model}未配置API KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 优化提示词，要求模型返回规范的JSON格式
    data_text = ""
    for key, value in data_dict.items():
        data_text += f"\n{key}: {json.dumps(value, ensure_ascii=False, indent=2)}\n"
    
    full_prompt = f"""{prompt}

数据内容：{data_text}

请注意：
1. 请直接返回处理后的JSON结果，不要包含其他解释文字
2. 结果必须是有效的JSON数组格式
3. 如果需要解释，请在JSON结果后面添加说明

JSON结果："""

    data = {
        "inputs": {},  # 清空 inputs
        "query": full_prompt,  # 将数据包含在 query 中
        "response_mode": "blocking",
        "user": user_id,
        "conversation_id": conversation_id
    }
    
    # 添加调试信息
    print("=== 调试信息 ===")
    print("请求数据:", json.dumps(data, ensure_ascii=False, indent=2))
    print("原始数据字典:", data_dict)
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
                # 尝试解析返回的JSON结果
                try:
                    answer_text = result["answer"]
                    processed_data = None
                    
                    # 方法1: 尝试直接解析整个回答
                    try:
                        if answer_text.strip().startswith('[') or answer_text.strip().startswith('{'):
                            processed_data = json.loads(answer_text.strip())
                    except json.JSONDecodeError:
                        pass
                    
                    # 方法2: 如果方法1失败，尝试从文本中提取JSON块
                    if processed_data is None:
                        processed_data = extract_json_from_text(answer_text)
                    
                    return {
                        "answer": answer_text,
                        "processed_data": processed_data
                    }
                except Exception as e:
                    return {
                        "answer": result["answer"],
                        "processed_data": None
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

# 统一数据处理接口
@app.post("/data-process/execute", response_model=NewDataProcessResponse)
async def execute_data_process(request: Dict[str, Any]) -> NewDataProcessResponse:
    try:
        # 提取基本参数
        model = request.get("model")
        user_prompt = request.get("user_prompt")
        
        if not model or not user_prompt:
            return NewDataProcessResponse(
                status="error",
                error_details="缺少必要参数: model 和 user_prompt"
            )

        data_dict = {}
        for key, value in request.items():
            if key.startswith('data') and key[4:].isdigit():
                data_dict[key] = value

        if not data_dict:
            return NewDataProcessResponse(
                status="error",
                error_details="未找到任何数据字段 (data0, data1, ...)"
            )

        # 调用 Dify 处理数据
        result = await call_dify_data_tool(
            model=model,
            prompt=user_prompt,
            data_dict=data_dict,
            user_id=request.get("user_id", "defaultid")
        )

        # 提取处理结果
        processed_data = result.get("processed_data")
        
        if processed_data is None:
            # 如果无法解析JSON，尝试其他方式或返回错误
            return NewDataProcessResponse(
                status="error",
                error_details=f"无法解析处理结果为JSON格式。原始回答: {result.get('answer', '')}"
            )

        return NewDataProcessResponse(
            status="success",
            result=processed_data
        )

    except Exception as e:
        tb = traceback.format_exc()
        return NewDataProcessResponse(
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
