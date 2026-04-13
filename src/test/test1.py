import requests

url = "http://127.0.0.1:8100/v1/chat/completions"
payload = {
    "model": "Qwen3-8B",
    "messages": [
        {"role": "system", "content": """
         任务：
请根据下面“职业细类”的任职要求样本，整理该职业细类的技能词典。

抽取要求：
1. 只保留可标准化、可复用、可统计的技能词。
2. 排除软素质、人格特质、福利待遇、年龄学历年限、岗位名称、空泛职责动词。
3. 同义词、缩写、大小写变体请收敛到同一个标准技能词下。
4. 尽量输出高精度词典，不要为了召回率加入模糊词。
5. 若某技能明显属于“编程语言/框架/数据库/工具/办公软件/证书/行业工具”等，请写明 skill_type。

         """},
        {"role": "user", "content": "请提取这段JD中的硬技能：职责描述： 1、精通 Java语言及 Java EE相关技术，熟练掌握 Java Web编程；熟悉java设计模式，有一定的前端能力； 2、掌握Spring/SpringMvc/MyBatis等开源框架； 3、熟练掌握 MySQL的使用和 SQL优化，熟练掌握redis的使用； 4、具备良好的编程习惯，能够编写高质量技术文档； 5、有分布式系统开发经验，对duboo、Spring Boot微服务、消息服务、负载均衡、高可用机制等有一定的理解； 6、有强烈的责任心，主动性强，良好的沟通表达能力和团队协作能力； 任职要求： 1、2年以上工作经验，有大型互联网项目经验优先； 2、统招本科以上学历。"},
    ],
    "temperature": 0.2,
    "max_tokens": 5120,
}

resp = requests.post(url, json=payload, timeout=120)
resp.raise_for_status()
data = resp.json()
print(data["choices"][0]["message"]["content"])
