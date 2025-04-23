import asyncio
import openai
from openai import AsyncOpenAI
from httpx import Timeout

async def test_async_openai_connection():
    try:
        client = AsyncOpenAI(
            api_key="sk-EhcEaokF8VPqmDKKZZaMYj3KOHCK1tskU947ov5VyRuH2M1B",  # ← 换成你的
            base_url="https://api2.aigcbest.top/v1",  # ❗️不要加路径！
            timeout=Timeout(60.0),
        )

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Ping"}],
        )

        print("✅ Connection successful! Response:\n")
        print(response)

    except Exception as e:
        print("❌ Connection failed!")
        print(e)

if __name__ == "__main__":
    asyncio.run(test_async_openai_connection())
