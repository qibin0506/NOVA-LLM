<div align="center">
    <img alt="created by gemini" src="./images/nova.jpeg" style="width: 50%">
</div>

<div align="center">
  <h1>N.O.V.A LLM</h1>
</div>

1. 命名：N.O.V.A，取自经典手游《[Near Orbit Vanguard Alliance](https://en.wikipedia.org/wiki/N.O.V.A._Near_Orbit_Vanguard_Alliance)》
2. 目标：实现单机PC可推理，个人可训练的LLM。
3. 特性：提供完成web推理应用程序，开箱即用；提供数据预处理、预训练、sft、dpo、grpo完整代码，开箱即训。

---

<div align="center">
  <img src="./images/web_ui.png">
</div>

### 准备工作
1. 安装依赖
    ```
    pip3 install -r requirements.txt
    ```
2. 下载LLM模型库并安装 [llm_model](https://github.com/qibin0506/llm-model-pytorch/releases/download/llm_model/project_llm_model-0.1-py3-none-any.whl)
   ```
   pip3 install project_llm_model-0.1-py3-none-any.whl
   ```
3. 下载训练库并安装 [llm_trainer](https://github.com/qibin0506/llm_trainer/releases/download/llm_trainer/project_llm_trainer-0.1-py3-none-any.whl)
   ```
   pip3 install project_llm_trainer-0.1-py3-none-any.whl
   ```
4. 下载web ui依赖并将文件解压到static目录 [web_ui.zip](https://github.com/qibin0506/NOVA-LLM/releases/download/dependence/web_ui.zip)
   
### 推理
1. 下载模型权重并将权重文件放到modeling目录 [dpo.bin](https://github.com/qibin0506/NOVA-LLM/releases/download/dependence/dpo.bin)、[reasoning.bin](https://github.com/qibin0506/NOVA-LLM/releases/download/dependence/reasoning.bin)、[grpo.bin](https://github.com/qibin0506/NOVA-LLM/releases/download/dependence/grpo.bin)
2. 运行app.py，然后浏览器访问 [http://localhost:8080](http://localhost:8080)
    ```
    python3 app.py
    ```

### 训练
训练流程可以按照如下顺序进行处理：
处理数据 -> 预训练 -> SFT -> DPO（可选，因模型太小，效果不明显） -> Reasoning -> GRPO（可选，因模型太小，效果不明显）

#### 数据预处理
#### 预训练
#### SFT
#### DPO
#### Reasoning
#### GRPO
