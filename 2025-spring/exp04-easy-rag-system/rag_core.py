import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from config import MAX_NEW_TOKENS_GEN, TEMPERATURE, TOP_P

# 自动加载 .env 文件中的环境变量
load_dotenv()

# 初始化 OpenAI 客户端 (Qwen API)
# 尝试从 QWEN_KEY 或 DASHSCOPE_API_KEY 获取
api_key = os.getenv("QWEN_KEY") or os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    print("\n" + "!"*50)
    print("❌ 错误: 未找到 API Key!")
    print("请在项目根目录创建 .env 文件并添加: QWEN_KEY=你的密钥")
    print("或者在终端设置环境变量: set QWEN_KEY=你的密钥 (CMD) 或 $env:QWEN_KEY='你的密钥' (PowerShell)")
    print("!"*50 + "\n")

client = OpenAI(
    api_key=api_key if api_key else "empty",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=120.0, # 进一步增加超时时间到 120 秒，应对网络波动
)

def generate_answer_stream(query, context_docs, max_retries=3):
    """Generates an answer stream using Qwen API based on query and context."""
    if not context_docs:
        yield {"text": "未找到相关文档以回答您的问题。", "token_count": 0, "speed": 0, "elapsed": 0}
        return

    context = "\n\n---\n\n".join([doc.get('content', '') for doc in context_docs])
    messages = [
        {"role": "system", "content": "你是一个专业的医疗助手。请仅根据提供的参考文档回答问题。如果文档中没有相关信息，请明确回答“在参考文档中未找到相关信息”。"},
        {"role": "user", "content": f"参考文档：\n{context}\n\n用户问题：{query}"}
    ]
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model="qwen-turbo-latest", 
                messages=messages,
                stream=True,
                max_tokens=MAX_NEW_TOKENS_GEN,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            
            token_count = 0
            yielded_any = False
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    new_text = chunk.choices[0].delta.content
                    yielded_any = True
                    token_count += 1 
                    elapsed = time.time() - start_time
                    speed = token_count / elapsed if elapsed > 0 else 0
                    yield {
                        "text": new_text,
                        "token_count": token_count,
                        "speed": round(speed, 2),
                        "elapsed": round(elapsed, 2)
                    }
            
            if not yielded_any:
                yield {"text": "模型未生成任何内容，请尝试重新提问。", "token_count": 0, "speed": 0, "elapsed": 0}
            
            return
                
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            print(f"Attempt {retry_count} failed: {error_msg}")
            
            if retry_count < max_retries:
                time.sleep(2) # 增加等待时间
                continue
            
            error_detail = "请求超时，请检查网络连接或稍后再试。" if "timeout" in error_msg.lower() else error_msg
            yield {"text": f"抱歉，调用 Qwen API 时出错 (已重试 {max_retries} 次): {error_detail}", "token_count": 0, "speed": 0, "elapsed": 0}
            break

def generate_answer(query, context_docs):
    """Generates an answer using Qwen API based on query and context."""
    if not context_docs:
        return {"answer": "未找到相关文档以回答您的问题。", "stats": {"token_count": 0, "speed": 0, "elapsed": 0}}

    context = "\n\n---\n\n".join([doc.get('content', '') for doc in context_docs])
    messages = [
        {"role": "system", "content": "你是一个专业的医疗助手。请仅根据提供的参考文档回答问题。如果文档中没有相关信息，请明确回答“在参考文档中未找到相关信息”。"},
        {"role": "user", "content": f"参考文档：\n{context}\n\n用户问题：{query}"}
    ]
    
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model="qwen-turbo-latest",
            messages=messages,
            stream=False,
            max_tokens=MAX_NEW_TOKENS_GEN,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        
        elapsed = time.time() - start_time
        answer = response.choices[0].message.content
        token_count = response.usage.total_tokens if hasattr(response, 'usage') else 0
        speed = token_count / elapsed if elapsed > 0 else 0
        
        return {
            "answer": answer.strip(),
            "stats": {
                "token_count": token_count,
                "speed": round(speed, 2),
                "elapsed": round(elapsed, 2)
            }
        }
    except Exception as e:
        print(f"Error during Qwen API generation: {e}")
        return {"answer": f"抱歉，调用 Qwen API 时出错: {str(e)}", "stats": {"token_count": 0, "speed": 0, "elapsed": 0}}
