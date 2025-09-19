from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import httpx
import asyncio
from consul_utils import register_service, deregister_service
from config import SERVICE_NAME
from WorkflowValidator import SimplifiedWorkflowValidator
from call_llm import call_dify

app = FastAPI()

# 请求模型
class WorkflowGenerationRequest(BaseModel):
    model: str                          # 模型名称
    requestId: str                      # 请求ID
    user_id: Optional[str] = None       # 用户ID
    conversation_id: Optional[str] = None # 对话ID
    user_prompt: str                    # 用户需求描述
    template_type: Optional[str] = "data_processing" # 模板类型
    service_type: Optional[str] = "ml"  # 服务类型
    isWorkFlow: bool = True             # 设为 true

# 响应模型
class AsyncWorkflowResponse(BaseModel):
    requestId: str
    status: str  # "processing", "completed", "failed"
    message: str

# 工作流结果模型
class WorkflowResult(BaseModel):
    requestId: str
    status: str  # "success" or "error"
    conversation_id: Optional[str] = None
    workflow_info: Optional[Dict[str, Any]] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None

# 用于存储工作流结果的内存缓存
workflow_results_cache: Dict[str, WorkflowResult] = {}

# 更新后的模型定义
class WorkflowInfo(BaseModel):
    userId: str

class SimpleAttribute(BaseModel):
    name: str
    value: str
    valueType: str

class ComplicatedAttribute(BaseModel):
    name: str
    value: Dict[str, Any]

class SourceAnchor(BaseModel):
    nodeName: str
    nodeMark: int
    seq: int = 0  # 新增序号字段

class TargetAnchor(BaseModel):
    nodeName: str
    nodeMark: int
    seq: int = 0  # 新增序号字段

class InputAnchor(BaseModel):
    seq: int = 0  # 新增锚点序号
    numOfConnectedEdges: int = 0
    sourceAnchor: Optional[SourceAnchor] = None

class OutputAnchor(BaseModel):
    seq: int = 0  # 新增锚点序号
    numOfConnectedEdges: int = 0
    targetAnchors: List[TargetAnchor] = []

class Node(BaseModel):
    id: str
    name: str
    mark: str                           # 组件唯一标识
    position: List[int]
    simpleAttributes: List[SimpleAttribute] = []
    complicatedAttributes: List[ComplicatedAttribute] = []
    inputAnchors: List[InputAnchor] = []
    outputAnchors: List[OutputAnchor] = []

@app.on_event("startup")
async def startup_event():
    """服务启动时注册到Consul"""
    SERVICE_PORT = 8004
    service_id = register_service(SERVICE_PORT)
    if service_id:
        app.state.service_id = service_id
        print(f"workflow_generator服务已注册到Consul，服务ID: {service_id}")

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时从Consul注销"""
    if hasattr(app.state, 'service_id'):
        deregister_service(app.state.service_id)

@app.post("/workflow/generate", response_model=AsyncWorkflowResponse)
async def generate_workflow(request: WorkflowGenerationRequest):
    try:
        asyncio.ensure_future(process_workflow_generation(request))
        
        return AsyncWorkflowResponse(
            requestId=request.requestId,
            status="processing",
            message="工作流生成任务已提交，正在处理中..."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提交工作流生成任务失败: {str(e)}")

async def process_workflow_generation(request: WorkflowGenerationRequest):
    try:
        # 1. 调用大模型生成工作流
        llm_response, new_conversation_id = await call_dify_with_workflow(
            model=request.model,
            prompt=request.user_prompt,
            user_id=request.user_id or "anonymous",
            conversation_id=request.conversation_id or "",
            request_id=request.requestId,
            isWorkFlow=str(request.isWorkFlow).lower()
        )
        
        # 2. 解析LLM响应
        workflow_structure = parse_llm_response(
            llm_response=llm_response,
            user_id=request.user_id,
            service_type=request.service_type,
            request_id=request.requestId,
            conversation_id=new_conversation_id
        )
        
        # 3. 工作流校验
        validator = SimplifiedWorkflowValidator()
        sanitized_workflow, warnings, errors = validator.sanitize(workflow_structure)
        
        if sanitized_workflow is None:
            # 发送失败结果给Java
            error_result = WorkflowResult(
                requestId=request.requestId,
                status="error",
                error_message=f"工作流结构校验失败: {', '.join(errors)}"
            )
            
            # 缓存失败结果用于测试接口
            workflow_results_cache[request.requestId] = error_result
            
            await send_result_to_java(error_result)
            return
        
        if warnings:
            print(f"工作流校验警告: {', '.join(warnings)}")
        
        # 4. 发送成功结果给Java
        result = WorkflowResult(
            requestId=request.requestId,
            status="success",
            conversation_id=new_conversation_id,
            workflow_info=sanitized_workflow["workflow_info"],
            nodes=sanitized_workflow["nodes"]
        )
        
        # 缓存结果用于测试接口
        workflow_results_cache[request.requestId] = result
        
        await send_result_to_java(result)
        
    except Exception as e:
        # 发送失败结果给Java
        error_result = WorkflowResult(
            requestId=request.requestId,
            status="error",
            error_message=f"工作流生成失败: {str(e)}"
        )
        
        # 缓存失败结果用于测试接口
        workflow_results_cache[request.requestId] = error_result
        
        await send_result_to_java(error_result)

async def send_result_to_java(result: WorkflowResult):
    try:
        #工作流回调接口
        callback_url = "http://localhost:7003/llm/result/experiment"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        timeout = httpx.Timeout(30.0, read=30.0, connect=10.0)
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                callback_url,
                headers=headers,
                json=result.dict()
            )
            
            if response.status_code == 200:
                print(f"成功发送结果给Java端，requestId: {result.requestId}")
            else:
                print(f"发送结果给Java端失败，状态码: {response.status_code}, 响应: {response.text}")
                
    except httpx.RequestError as e:
        print(f"发送结果给Java端请求失败: {e}")
    except Exception as e:
        print(f"发送结果给Java端时发生未知错误: {e}")

# dify调用函数
async def call_dify_with_workflow(model: str, prompt: str, user_id: str, request_id: str, 
                                 conversation_id: Optional[str] = None, isWorkFlow: str = "false") -> tuple:
    try:
        from call_llm import MODEL_TO_APIKEY, dify_url
        import httpx
        
        api_key = MODEL_TO_APIKEY.get(model)
        if not api_key:
            raise ValueError(f"模型{model}未配置API KEY")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "inputs": {
                "requestId": request_id,
                "isWorkFlow": isWorkFlow
            },
            "query": prompt,
            "response_mode": "blocking",
            "user": user_id,
            "conversation_id": conversation_id or ""
        }

        timeout = httpx.Timeout(120.0, read=120.0, connect=10.0)

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

def parse_llm_response(llm_response: Any, user_id: str, service_type: str, request_id: str, conversation_id: str = None) -> Dict[str, Any]:
    try:
        if isinstance(llm_response, tuple) and len(llm_response) >= 1:
            response_data = llm_response[0]
        else:
            response_data = llm_response
        
        if not isinstance(response_data, str):
            response_data = str(response_data)
        
        print(f"解析的响应数据: {response_data[:500]}...")
        
        start_idx = response_data.find('{')
        end_idx = response_data.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response_data[start_idx:end_idx]
            print(f"提取的JSON字符串: {json_str[:200]}...")
            
            workflow_data = json.loads(json_str)
            
            if not isinstance(workflow_data, dict):
                raise ValueError(f"解析的工作流数据不是字典类型，而是: {type(workflow_data)}")
            
            if "workflow_info" not in workflow_data:
                workflow_data["workflow_info"] = {}
            
            if "nodes" not in workflow_data:
                workflow_data["nodes"] = []
            
            if not isinstance(workflow_data["nodes"], list):
                raise ValueError(f"nodes字段不是列表类型，而是: {type(workflow_data['nodes'])}")
            
            workflow_data["requestId"] = request_id
            workflow_data["conversation_id"] = conversation_id
            
            if not isinstance(workflow_data["workflow_info"], dict):
                workflow_data["workflow_info"] = {}
            workflow_data["workflow_info"]["userId"] = user_id or "anonymous"
            
            # 处理节点数据，适配新格式
            for i, node in enumerate(workflow_data["nodes"]):
                if not isinstance(node, dict):
                    raise ValueError(f"节点{i}不是字典类型，而是: {type(node)}")
                
                # 确保必要字段存在
                if "id" not in node:
                    node["id"] = f"node_{i}"
                
                if "mark" not in node:
                    node["mark"] = node.get("id", f"node_{i}")
                
                if "position" not in node:
                    node["position"] = [100 + i * 200, 100]
                
                # 处理position字段格式
                if isinstance(node["position"], dict):
                    if "x" in node["position"] and "y" in node["position"]:
                        node["position"] = [node["position"]["x"], node["position"]["y"]]
                
                if "name" not in node:
                    node["name"] = node.get("id", f"node_{i}")
                
                # 初始化属性列表
                node.setdefault("simpleAttributes", [])
                node.setdefault("complicatedAttributes", [])
                node.setdefault("inputAnchors", [])
                node.setdefault("outputAnchors", [])
                
                # 确保锚点是列表类型
                if not isinstance(node["inputAnchors"], list):
                    node["inputAnchors"] = []
                if not isinstance(node["outputAnchors"], list):
                    node["outputAnchors"] = []
                
                # 处理inputAnchors - 适配新格式
                for j, input_anchor in enumerate(node["inputAnchors"]):
                    if isinstance(input_anchor, dict):
                        # 添加seq字段
                        input_anchor.setdefault("seq", j)
                        input_anchor.setdefault("numOfConnectedEdges", 0)
                        
                        # 处理旧格式的sourceAnchors转换为新格式的sourceAnchor
                        if "sourceAnchors" in input_anchor and input_anchor["sourceAnchors"]:
                            if len(input_anchor["sourceAnchors"]) > 0:
                                old_source = input_anchor["sourceAnchors"][0]
                                input_anchor["sourceAnchor"] = {
                                    "nodeName": old_source.get("nodeName", ""),
                                    "nodeMark": old_source.get("nodeMark", 0),
                                    "seq": old_source.get("seq", 0)  # 添加seq字段
                                }
                            # 移除旧字段
                            input_anchor.pop("sourceAnchors", None)
                        
                        # 确保sourceAnchor包含seq字段
                        if "sourceAnchor" in input_anchor and input_anchor["sourceAnchor"]:
                            if "seq" not in input_anchor["sourceAnchor"]:
                                input_anchor["sourceAnchor"]["seq"] = 0
                
                # 处理outputAnchors - 适配新格式
                for j, output_anchor in enumerate(node["outputAnchors"]):
                    if isinstance(output_anchor, dict):
                        # 添加seq字段
                        output_anchor.setdefault("seq", j)
                        output_anchor.setdefault("numOfConnectedEdges", 0)
                        output_anchor.setdefault("targetAnchors", [])
                        
                        # 确保targetAnchors中的每个元素都有正确的格式和seq字段
                        for k, target_anchor in enumerate(output_anchor["targetAnchors"]):
                            if isinstance(target_anchor, dict):
                                target_anchor.setdefault("nodeName", "")
                                target_anchor.setdefault("nodeMark", 0)
                                target_anchor.setdefault("seq", k)  # 添加seq字段
            
            return workflow_data
        else:
            raise ValueError("LLM响应中未找到有效的JSON结构")
            
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON解析失败: {e}")
    except Exception as e:
        raise ValueError(f"解析LLM响应失败: {e}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "workflow_generator"}

@app.get("/workflow/result/{request_id}")
async def get_workflow_result(request_id: str):
    """获取工作流生成结果"""
    if request_id not in workflow_results_cache:
        raise HTTPException(status_code=404, detail=f"未找到请求ID为 {request_id} 的工作流结果")
    
    return workflow_results_cache[request_id]

@app.get("/workflow/results")
async def get_all_workflow_results():
    """获取所有工作流生成结果（用于测试）"""
    return {
        "total": len(workflow_results_cache),
        "results": workflow_results_cache
    }

@app.delete("/workflow/result/{request_id}")
async def delete_workflow_result(request_id: str):
    """删除指定的工作流结果"""
    if request_id not in workflow_results_cache:
        raise HTTPException(status_code=404, detail=f"未找到请求ID为 {request_id} 的工作流结果")
    
    del workflow_results_cache[request_id]
    return {"message": f"已删除请求ID为 {request_id} 的工作流结果"}

@app.delete("/workflow/results")
async def clear_all_workflow_results():
    """清空所有工作流结果缓存（用于测试）"""
    count = len(workflow_results_cache)
    workflow_results_cache.clear()
    return {"message": f"已清空所有工作流结果缓存，共删除 {count} 条记录"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)