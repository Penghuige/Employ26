import os
from openai import OpenAI, BadRequestError

client = OpenAI(
    api_key=os.getenv("OPENAIE_API_KEY"), # 如果您没有配置环境变量，请在此处替换您的API-KEY
    base_url="https://www.packyapi.com/v1",
)

try:
    completion = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '请根据以下岗位文本，提取出岗位名称、职业中类、职业编码等信息，并以JSON格式输出。'},
        
            {'role': 'system', 'content': 
            """
样本ID: recruit.main.gd_recruit_liepin_sample:17 
岗位名称: 高级/资深招聘HR（游戏板块）
职业中类: 粮油竞价交易员
职业编码: 4-01-03-02

岗位文本：1、本科及以上学历，3年以上互联网/游戏行业招聘经验，热爱游戏为佳 2、熟
练掌握各种招聘渠道的应用，熟悉招聘全流程工作 3、思维敏捷，善于沟通，有良好的自 
驱力，目标导向 4、心态积极乐观，抗压性好，乐于合作

样本ID: recruit.main.gd_recruit_liepin_sample:15
岗位名称: 摄像
职业中类: 摄影记者
职业编码: 2-10-01-02

岗位文本：大学本科以上学历，具备新闻拍摄编辑工作相关从业经验 | 熟悉视频新闻内容
生产特点，具备视频制作、包装等能力 | 熟练使用佳能、索尼等摄像机并进行拍摄，具有
一定的航拍经验优先 | 能适应高强度工作压力，具有团队合作精神。

样本ID: recruit.main.gd_recruit_liepin_sample:4
岗位名称: 高级研究员/科学家
职业中类: 疾病控制医师
职业编码: 2-05-05-01

岗位文本：病毒学、生物学、医学、药学等相关专业博士学历 | 具有较为深厚的病毒学、
免疫学和分子生物学知识基础 | 具有AAV或其它类型病毒相关项目经验，或多个项目成功 
经验者优先 | 具有较强的学习能力，能独立查阅文献和分析解决问题的能力 | 有团队合 
作精神和良好的沟通能力，工作态度认真负责。

样本ID: recruit.main.gd_recruit_liepin_sample:12
岗位名称: 饲料研发助理
职业中类: 兽医
职业编码: 2-03-06-01

岗位文本：1.全日制硕士及以上学历，动物科学、动物营养、水产、中兽医、微生物发酵 
相关专业。 2. | 有优秀的专业基础和丰富的专业知识，良好的创新力、内驱力和执行力 
。

样本ID: recruit.main.gd_recruit_liepin_sample:3
岗位名称: 资深电源工程师
职业中类: 电工电器工程技术人员
职业编码: 2-02-11-01

岗位文本：本科及以上学历，电子类相关专业 | 8年以上AC/DC、DC/DC、PFC等模块电源及
UPS、逆变器、车载充电机等电源产品开发设计经验，能独立完成大功率电源、多路电源组
件、电源系统产品设计工作 | 具有一定的英语水平，能够阅读英文器件手册 | 精通反激 
、正激、LLC、全桥、同步整流等电路的工作原理 | 精通电源产品可靠性设计规范，熟悉 
电路设计、电路测试与调试、PCB板设计、安全设计、热设计、防护设计、电磁兼容设计等
电源设计 | 熟悉PADS、Altium Designer、Protel99SE、AutoCAD等绘图软件，熟悉Saber 
等仿真软件。 7、能熟练使用各种测试仪器仪表。 8、具有全砖、半砖、1/4砖等标准砖型
电源模块开发经验者优先 | 具有大功率AC/DC、DC/DC、多路电源组件、电源系统等产品开
发经验者优先 | 具有数字电源开发经验者优先 | 责任心强，具有良好的抗压能力和团队 
合作精神。

样本ID: recruit.main.gd_recruit_liepin_sample:22
岗位名称: 项目总经理
职业中类: 建筑和市政设计工程技术人员L
职业编码: 2-02-18-01

岗位文本：具备丰富的行业理论与实践知识以及全面的企业管理知识 | 熟悉国家颁布的各
项技术标准、规程，了解国内外建筑领域的新技术、新方法、新知识，全面掌握房地产开 
发的业务流程 | 较强的领导、组织、协调和沟通能力。

样本ID: recruit.main.gd_recruit_liepin_sample:14
岗位名称: 高级java工程师
职业中类: 快递工程技术人员
职业编码: 2-02-13-02

岗位文本：写在前面的话:有微服务、分布式(springcloud)、其他组还需要Netty(Java一 
种开源框架)、CI/CD(持续集成、持续交付构建工具)。项目经验有B端经验会更好哦，加油
加油加油。 | 本科及以上学历，计算机相关专业毕业，3年以上Java软件开发经验 | 熟练
掌握springboot/mybatis/springcloud/redis/mq等 | 熟练使用MySQL/mongoDB数据库 |  
具备分布式、高可用系统架构设计能力和相关经验；熟悉微服务框架、分布式存储、搜索 
、异步框架、集群与负载均衡，消息中间件、分库分表等技术 | 业务理解能力强，熟练掌
握软件设计原则，业务架构能力强 | 具备良好的主动性、执行力、团队协作能力 | 有ERP/MES/APS 等系统开发经验优先。

样本ID: recruit.main.gd_recruit_liepin_sample:8
岗位名称: 0342WI-数据挖掘岗
职业中类: 统计专业人员
职业编码: 2-06-02-00

岗位文本：重点大学研究生以上学历，统计/精算/保险/数学/计算机等相关专业 | 对数据
分析建模整套流程有较全面深入的理解 | 熟练使用hive、spark、presto、python、SAS等
数据分析工具 | 熟悉常用数据挖掘、机器学习算法，有建模工作经验者优先。

样本ID: recruit.main.gd_recruit_liepin_sample:26
岗位名称: 有机检测工程师
职业中类: 临床检验技师
职业编码: 2-05-07-04

岗位文本：专科及以上学历，环境监测、分析化学、环境工程、药物分析等与环境检测、 
职业卫生相关专业优先 | 动手能力强，熟悉实验室分析基础知识，常用分析方法的原理和
操作，常用仪器设备的原理和操作优先 | 熟练掌握岛津气相色谱质谱，并熟练使用相应工
作站进行数据采集和数据分析 | 能熟练操作色谱分析仪器:如GC/GCMS/LC/LCMSMS者优先 | 独立完成过能力验证优先考虑 | 工作认真，吃苦耐劳，能适应公司加班要求 | 具有较强
的责任感和团队合作精神。

样本ID: recruit.main.gd_recruit_liepin_sample:16
岗位名称: 海外内容安全（英语）
职业中类: 翻译
职业编码: 2-10-05-01

"""},
            ],
        # 本代码示例均采用流式输出，以清晰和直观地展示模型输出过程。如果您希望查看非流式输出的案例，请参见https://help.aliyun.com/zh/model-studio/text-generation
        stream=True,
        stream_options={"include_usage": True}
    )

    full_content = ""
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content
            print(chunk.model_dump())
    
    print(full_content)

except BadRequestError as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")