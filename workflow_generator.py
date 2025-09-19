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

    # 方案1：将数据直接包含在 query 中
    data_text = ""
    for key, value in data_dict.items():
        data_text += f"\n{key}: {json.dumps(value, ensure_ascii=False, indent=2)}\n"
    
    full_prompt = f"{prompt}\n\n数据内容：{data_text}"

    data = {
        "inputs": {},  # 清空 inputs
        "query": full_prompt,  # 将数据包含在 query 中
        "response_mode": "blocking",
        "user": user_id,
        "conversation_id": conversation_id
    }

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
                    # 尝试解析为JSON
                    if answer_text.strip().startswith('[') or answer_text.strip().startswith('{'):
                        processed_data = json.loads(answer_text)
                    else:
                        # 如果不是JSON格式，可能需要从文本中提取
                        processed_data = None
                    
                    return {
                        "answer": answer_text,
                        "processed_data": processed_data
                    }
                except json.JSONDecodeError:
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
