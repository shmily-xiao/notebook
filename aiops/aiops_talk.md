回顾AIOps的发展历程

涉及到哪些环节和组件。

每一个环节有哪些处理的方式，软件和case （裴丹老师的课程里面有）。 有哪些算法和策略。


机器学习？ 大模型？

现在大模型是否能够重新定义AIOps。 

行业案例。
sunfire 的AIOps 的案例实践。
谷歌的实践案例。 我们和他的区别是什么。


我们是怎么做的？



什么是AIOps，是智能化运维？ 那智能化运维，里面又有什么？ 怎么去构建一个智能化运维体系呢？

现在遇到的困境和挑战是什么？


从故障开始到结束主要有四大核心能力，即故障发现、告警触达、故障定位、故障恢复


故障发现 包含了指标预测、异常检测和故障预测等方面，主要目标是能及时、准确地发现故障；
告警触达 包含了告警事件的收敛、聚合和抑制，主要目标是降噪聚合，减少干扰；
故障定位 包含了数据收集、根因分析、关联分析、智能分析等，主要目标是能及时、精准地定位故障根因；
故障恢复 部分包含了流量切换、预案、降级等，主要目标是及时恢复故障，减少业务损失。


1 - 5 - 10
1分钟 发现 5分钟定位 10分钟恢复



1.1 什么是AIOps？
定义解析
AIOps，全称Artificial Intelligence for IT Operations，是Gartner在2016年提出的概念。它是一种结合大数据和机器学习功能的平台，用于增强和部分替代广泛的IT运维流程和任务。
核心理念
● 数据驱动: 基于海量运维数据进行智能分析
● 自动化: 从反应式运维转向预测式和自愈式运维
● 智能化: 利用AI算法提升运维决策的准确性和效率
技术演进历程
传统运维 → 自动化运维 → DevOps → AIOps
   ↓           ↓          ↓       ↓
人工处理    脚本自动化   流程整合  智能预测

AIops 不等于大模型


集团是怎么做aiops的？
 
 集团没有一个统一的aiops平台，这些功能散落在不同的系统中。
 sunfire、鹰眼、asi、appinsight、schedulerX、诺曼底等等。
 ··· todo 需要尽量去了解现有集团的运维平台，比如星环啊等等。

speed

 还有ai的运维平台， mlflow、

故障发现 metric log event 

告警触达

故障定位

故障恢复

每一个阶段的开源的产品。和算法。



附件

大语言模型时代的智能运维（AIOps）综述：https://zhuanlan.zhihu.com/p/1931137819994199060

AIOps在美团的探索与实践——故障发现篇：https://zhuanlan.zhihu.com/p/266766105

一、浅谈智能运维AIOps概述：https://zhuanlan.zhihu.com/p/670668732

提高IT运维效率，深度解读京东云AIOps落地实践（异常检测篇）： https://zhuanlan.zhihu.com/p/1908923139712357410

AIOps explained: Applying AI to predict and prevent outages: https://www.youtube.com/watch?v=8kNwcrpRsGk&list=PL_4RxtD-BL5uuXsmuneBWS1oCo65Xe1On&index=16


Ai :https://en.m.wikipedia.org/wiki/Artificial_intelligence:

----
model realignment