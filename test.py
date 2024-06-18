from langchain_fireworks import Fireworks
from langchain_openai import OpenAI, ChatOpenAI
llm1 = Fireworks(
    model="accounts/fireworks/models/firefunction-v2",
    base_url="https://api.fireworks.ai/inference/v1/completions",
    max_tokens=1024
)
llm2 = OpenAI(model="gpt-4")
output = llm1.invoke("who is narendra modi")
print(output)