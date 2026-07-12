import argparse
import pandas as pd
from ollama import Client
import json
import re
import datetime
import os
import logging
import sys

def parse_list_arg(arg_str):
    """将逗号分隔的字符串转换为列表，并去除首尾空格"""
    if not arg_str:
        return []
    return [item.strip() for item in arg_str.split(',') if item.strip()]

def setup_logger(log_path, console_level=logging.WARNING, file_level=logging.INFO):
    """配置日志记录器：同时输出到文件和控制台"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 允许所有级别，由 handler 过滤

    # 清除已有的 handlers（避免重复）
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # 文件 handler
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(file_level)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # 控制台 handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def main():
    # ---------- 解析命令行参数 ----------
    parser = argparse.ArgumentParser(description="批量运行症状提取实验")
    parser.add_argument(
        "--models",
        type=str,
        default="gemma3:1b,gemma3:4b,llama3.2:3b,llama3.1:8b,deepseek-r1:7b,qwen3:1.7b,qwen3:8b",
        help="逗号分隔的模型名称列表"
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default="instruction0.txt,instruction1.txt,instruction2.txt",
        help="逗号分隔的提示词文件路径列表"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="D:/zhang/hxd/middata/zs200.csv",
        help="数据集路径"
    )
    parser.add_argument(
        "--num_symps",
        type=int,
        default=7,
        help="症状类别数量"
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="ZS",
        help="用于提取症状的文本列名"
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.0,0.2,0.4",
        help="逗号分隔的温度值列表"
    )
    parser.add_argument(
        "--top_ks",
        type=str,
        default="5,10,20",
        help="逗号分隔的 top_k 值列表"
    )
    parser.add_argument(
        "--top_ps",
        type=str,
        default="0.3,0.5,0.7",
        help="逗号分隔的 top_p 值列表"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./batch_results",
        help="输出根目录"
    )
    args = parser.parse_args()

    # 将参数转换为列表
    models = parse_list_arg(args.models)
    instructions = parse_list_arg(args.instructions)
    temperatures = parse_list_arg(args.temperatures)
    top_ks = [int(x) for x in parse_list_arg(args.top_ks)]  # top_k 需要整数
    top_ps = [float(x) for x in parse_list_arg(args.top_ps)]

    # 检查必要文件是否存在
    for inst in instructions:
        if not os.path.isfile(inst):
            print(f"警告：提示词文件 {inst} 不存在，请检查路径。")
    if not os.path.isfile(args.dataset):
        print(f"错误：数据集文件 {args.dataset} 不存在。")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建 Ollama 客户端（只创建一次）
    client = Client(host="http://localhost:11434", headers={"x-some-header": "some-value"})

    total_start = datetime.datetime.now()
    total_combinations = len(models) * len(instructions) * len(temperatures) * len(top_ks) * len(top_ps)
    combo_counter = 0

    # ---------- 遍历所有组合 ----------
    for model in models:
        for instruct_file in instructions:
            # 读取提示词内容
            try:
                with open(instruct_file, "r", encoding="utf8") as fp:
                    instruct = fp.read()
            except Exception as e:
                print(f"无法读取提示词文件 {instruct_file}：{e}")
                continue

            for temp in temperatures:
                for topk in top_ks:
                    for topp in top_ps:
                        combo_counter += 1
                        # 构造输出文件名（替换冒号等特殊字符）
                        model_safe = model.replace(':', '-')
                        instruct_base = os.path.splitext(os.path.basename(instruct_file))[0]
                        base_name = f"{model_safe}_{instruct_base}_temp{temp}_topk{topk}_topp{topp}"
                        csv_path = os.path.join(args.output_dir, f"result_{base_name}.csv")
                        log_path = os.path.join(args.output_dir, f"log_{base_name}.log")

                        # 配置日志
                        logger = setup_logger(log_path, console_level=logging.INFO, file_level=logging.DEBUG)
                        logger.info(f"===== 组合 {combo_counter}/{total_combinations} 开始 =====")
                        logger.info(f"模型: {model}")
                        logger.info(f"提示词文件: {instruct_file}")
                        logger.info(f"温度: {temp}, top_k: {topk}, top_p: {topp}")
                        logger.info(f"结果将保存至: {csv_path}")
                        logger.info(f"日志将保存至: {log_path}")

                        try:
                            # ---------- 每个组合独立加载数据 ----------
                            df = pd.read_csv(args.dataset)
                            # 初始化症状列
                            for col in range(args.num_symps):
                                df[col] = 0

                            # 遍历每个样本
                            for idx in range(len(df)):
                                content = df.loc[idx, args.feature]
                                logger.debug(f"处理样本 {idx}: {content[:50]}...")  # 只记录前50字符
                                try:
                                    response = client.chat(
                                        model=model,
                                        messages=[
                                            {"role": "system", "content": instruct},
                                            {"role": "user", "content": content},
                                        ],
                                        stream=False,
                                        options={
                                            "temperature": temp,
                                            "top_k": topk,
                                            "top_p": topp,
                                        },
                                    )
                                    # 解析 JSON
                                    jsonmatches = re.findall(r"```json\n{0,1}(.*?)\n{0,1}```", response.message.content, re.DOTALL)
                                    if jsonmatches:
                                        jsoncontent = jsonmatches[0]
                                        symp_ids = re.findall(r'"symp":\s{0,1}(\d+)', jsoncontent)
                                        if symp_ids:
                                            for sid in symp_ids:
                                                df.loc[idx, int(sid)] = 1
                                            logger.debug(f"样本 {idx} 更新症状: {symp_ids}")
                                        else:
                                            logger.warning(f"样本 {idx} 未找到症状ID")
                                    else:
                                        logger.warning(f"样本 {idx} 响应中未包含 JSON 代码块")
                                except Exception as e:
                                    logger.error(f"样本 {idx} 处理出错: {e}")

                            # 保存结果
                            df.to_csv(csv_path, index=False)
                            logger.info(f"结果已保存至 {csv_path}")

                        except Exception as e:
                            logger.error(f"组合 {combo_counter} 运行失败: {e}", exc_info=True)
                        else:
                            logger.info(f"组合 {combo_counter} 成功完成")
                        finally:
                            logger.info(f"===== 组合 {combo_counter} 结束 =====\n")

    # 总耗时
    total_end = datetime.datetime.now()
    elapsed = total_end - total_start
    print(f"所有组合运行完毕，总耗时: {elapsed}")

if __name__ == "__main__":
    main()