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

    # 将JSON数据保存为临时文件并上传
    files_data = []
    temp_file_urls = []
    
    try:
        # 创建主数据文件 - 包含所有数据集的信息
        short_uuid = str(uuid.uuid4())[:8]
        main_data_filename = f"{short_uuid}_all_data.json"
        main_data_path = os.path.join(SHARED_DIR, main_data_filename)
        
        # 将所有数据集合并到一个文件中，便于大模型理解完整的数据结构
        combined_data = {
            "datasets": data_dict,
            "dataset_count": len(data_dict),
            "dataset_names": list(data_dict.keys())
        }
        
        with open(main_data_path, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=2)
        
        main_data_url = f"http://10.92.64.224:8003/local_files/{main_data_filename}"
        temp_file_urls.append(main_data_path)
        
        files_data.append({
            "type": "document", 
            "transfer_method": "remote_url",
            "url": main_data_url,
            "upload_file_id": "combined_datasets"
        })
        
        # 同时在inputs中提供数据，供工具函数直接使用
        data_inputs = {}
        
        # 为单数据集操作提供第一个数据集
        if len(data_dict) >= 1:
            first_key = list(data_dict.keys())[0]
            data_inputs["file_content"] = json.dumps([data_dict[first_key]], ensure_ascii=False)
        
        # 为多数据集操作（如join）提供所有数据集
        if len(data_dict) >= 2:
            all_datasets = list(data_dict.values())
            data_inputs["file_content"] = json.dumps(all_datasets, ensure_ascii=False)

        data = {
            "inputs": data_inputs,  # 提供数据给工具函数使用
            "query": prompt,
            "files": files_data,  # 提供给大模型理解数据结构
            "response_mode": "blocking",
            "user": user_id,
            "conversation_id": conversation_id
        }

        print("=== 调试信息 ===")
        print("发送到Dify的数据:", json.dumps(data, ensure_ascii=False, indent=2))
        print("临时文件列表:", temp_file_urls)
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
    
    finally:
        # 清理临时文件
        for temp_file_path in temp_file_urls:
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    print(f"🗑️ 已删除临时文件: {temp_file_path}")
            except Exception as e:
                print(f"⚠️ 删除临时文件失败 {temp_file_path}: {e}")

# 统一的数据处理接口
@app.post("/data-process/execute", response_model=DataProcessResponse)
async def execute_data_process(request: Dict[str, Any]) -> DataProcessResponse:
    """
    统一的数据处理接口
    大模型会在Dify中自动选择合适的工具函数处理数据
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

        # 提取数据字段
        data_dict = {}
        for key, value in request.items():
            if key.startswith('data') and key[4:].isdigit():
                data_dict[key] = value

        if not data_dict:
            return DataProcessResponse(
                status="error",
                error_details="未找到任何数据字段 (data0, data1, ...)"
            )

        print(f"接收到数据处理请求: {len(data_dict)} 个数据集")
        for key, value in data_dict.items():
            print(f"- {key}: {len(value) if isinstance(value, list) else 'unknown'} 条记录")

        # 调用Dify，让大模型决策并调用工具函数
        result = await call_dify_with_tools(
            model=model,
            prompt=user_prompt,
            data_dict=data_dict,
            user_id=user_id
        )

        # 注意：实际的处理结果会通过工具函数返回
        # 这里需要从Dify的回答中解析出处理后的数据
        # 具体解析逻辑需要根据Dify工具函数的返回格式来定制
        
        return DataProcessResponse(
            status="success",
            answer=result.get("answer"),
            # result字段需要根据实际工具函数返回格式来解析
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
