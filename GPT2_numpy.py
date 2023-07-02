import numpy as np
import pickle

# 定义了GELU激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# 定义了softmax函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# 定义了Layer Normalization层
def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # 在最后一维上对x进行归一化，使其均值为0，方差为1
    return g * x + b  # 使用参数g（gamma）和b（beta）进行尺度变换和位置偏移

# 定义线性变换
def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b

# 定义前馈神经网络（Feed Forward Network，FFN）
def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # 首先是升维操作
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # 接着是降维操作
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x

# 定义自注意力机制
def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

# 定义多头注意力机制（Multi-head Attention）
def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # 对输入进行线性变换
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # 按最后一维将x分割为q, k, v
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # 按最后一维将q, k, v切分成n_head个头
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # 创建一个遮掩矩阵，用于在自注意力机制中隐藏未来的输入
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # 对每个头执行自注意力操作
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # 合并所有的头
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # 对输出进行线性变换
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x

# 定义Transformer模块
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # 首先是多头自注意力机制
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # 接着是前馈神经网络
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x

# GPT2模型的定义
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # 对输入进行词向量和位置向量的嵌入
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # 对输入进行n层的Transformer处理
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # 对结果进行投影以得到词汇分布
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]

# 定义文本生成函数
def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # 自回归的生成循环
        logits = gpt2(inputs, **params, n_head=n_head)  # 模型的前向计算
        next_id = np.argmax(logits[-1])  # 贪婪采样
        inputs.append(int(next_id))  # 将预测结果添加到输入中

    return inputs[len(inputs) - n_tokens_to_generate :]  # 只返回生成的ids

# 主函数
def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # 加载编码器、超参数和模型参数
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    save_path = r"Params\params.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(params, f)
    f.close()
    # 使用BPE tokenizer对输入字符串进行编码
    input_ids = encoder.encode(prompt)

    # 确保我们生成的序列长度不超过模型的最大长度
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # 生成输出id
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # 将id解码回字符串
    output_text = encoder.decode(output_ids)

    return output_text

def train(inputs, targets, params, n_head):
    # 前向传播
    logits = gpt2(inputs, **params, n_head=n_head)
    predictions = softmax(logits)

    # cross-entropy
    target_probs = np.eye(params['wte'].shape[0])[targets]  # one-hot encoding
    loss = -np.sum(target_probs * np.log(predictions)) / len(inputs)

    # 反向传播
    dlogits = predictions - target_probs
    dwte = dlogits.T @ logits[:-1]
    dlayers = dlogits @ params['wte'][:-1]

    # 更新参数
    learning_rate = 0.001
    params['wte'][:-1] -= learning_rate * dwte.T
    for block, dlayer in zip(params['blocks'], dlayers):
        block['ln_f']['b'] -= learning_rate * np.sum(dlayer, axis=0)  # Update layer norm bias
        block['ln_f']['g'] -= learning_rate * np.sum(dlayer * layer_norm(block['ln_2']['out'], **block['ln_f'], eps=1e-5), axis=0)  # Update layer norm scale
        block['ln_2']['b'] -= learning_rate * np.sum(dlayer * block['mlp']['out'], axis=0)  # Update layer norm bias
        block['ln_2']['g'] -= learning_rate * np.sum(dlayer * layer_norm(block['ln_1']['out'], **block['ln_2'], eps=1e-5), axis=0)  # Update layer norm scale
        block['mlp']['b'] -= learning_rate * np.sum(dlayer * block['ln_2']['out'], axis=0)  # Update MLP bias
        block['mlp']['w'] -= learning_rate * np.sum(dlayer[:, np.newaxis, :] * block['ln_2']['out'][:, :, np.newaxis], axis=0)  # Update MLP weights
        block['attn']['c_proj']['b'] -= learning_rate * np.sum(dlayer * block['ln_1']['out'], axis=0)  # Update attention projection bias
        block['attn']['c_proj']['w'] -= learning_rate * np.sum(dlayer[:, np.newaxis, :] * block['ln_1']['out'][:, :, np.newaxis], axis=0)  # Update attention projection weights
        block['attn']['c_attn']['b'] -= learning_rate * np.sum(dlayer * attention(block['ln_1']['out'], **block['attn'], n_head=n_head), axis=0)  # Update attention projection bias
        block['attn']['c_attn']['w'] -= learning_rate * np.sum(dlayer[:, np.newaxis, :] * block['ln_1']['out'][:, :, np.newaxis], axis=0)  # Update attention projection weights

    return loss


if __name__ == "__main__":
    import fire

    fire. Fire(main)
