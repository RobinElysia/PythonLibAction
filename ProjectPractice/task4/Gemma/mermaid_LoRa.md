1. 定义提示词模板
2. 加载数据（pd——》Dataset）
3. 加载Tokenizer
4. 处理数据（map + procFunc）
5. 加载基本模型（model）
6. 配置微调（config）
7. 将模型和微调配置合并（model）
8. 配置训练参数（arg）
9. 创建训练器对象，并传入参数（model，arg，train_dataset，data_collator）
10. 开始训练
11. 保存适配器（常见微调通用步骤）
    1. 常见的有config文件、tokenizer文件、模型文件、trainer_state文件
12. 合并参数（仅LoRa系微调）
13. 保存合并好的模型和Tokenizer（仅LoRa）