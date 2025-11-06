1. src/modelTrans.py为程序主要运行部件
2. 程序运行步骤详解：
  可以直接运行scripts/run.sh
  (也可以手动执行：程序运行要先下载相关环境pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch datasets sentencepiece scikit-learn matplotlib
  如果出现不能下载数据集的情况要运行export HF_ENDPOINT=https://hf-mirror.com
  之后就可以正常运行python src/modelTrans.py)
3. data/training_history.json是训练历史记录：记录每个epoch的训练损失和验证损失，用于分析模型训练过程和学习曲线，也可以重新绘制训练曲线或进行后续分析
4. data/training_curves.png 是训练曲线可视化：直观展示训练和验证损失的变化趋势，包含三个子图：训练损失和验证损失对比，单独的训练损失曲线，单独的验证损失曲线
5. data/vocab.pkl是词汇表文件：保存构建的词汇表，包含单词到索引的映射，确保训练和推理时使用相同的词汇表
6. data/ablation_results.json 是消融实验结果：记录不同模型配置的详细实验结果：比较不同消融实验的性能差异，包含模型配置参数、参数量、最佳验证损失、测试损失、每个epoch的训练/验证损失
7. data/ablation_study_results.png 是消融实验可视化：包含验证损失对比、测试损失对比、参数量对比、验证vs测试损失
8. data/attention_heads_comparison.png 是注意力头数分析，专门分析不同注意力头数对模型性能的影响
9. 运行要求：
   CPU: 4核以上，内存: 8GB RAM，存储: 10GB 可用空间，GPU无要求
