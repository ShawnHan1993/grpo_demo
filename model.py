import tensorflow as tf
keras = tf.keras
import math
import numpy as np

class ShapeChecker:
    def __init__(self):
        """
        初始化 ShapeChecker 类。
        """
        pass

    def __call__(self, tensor, expected_shape, name="tensor"):
        """
        检查张量的形状是否符合预期，支持动态形状（None 维度）。

        参数:
        - tensor: 要检查的张量 (tf.Tensor)。
        - expected_shape: 预期的形状 (tuple 或 list)，允许某些维度为 None。
        - name: 张量的名称，用于错误信息 (可选)。

        返回值:
        - 如果形状匹配，返回 True。
        - 如果形状不匹配，抛出 ValueError。
        """
        actual_shape = tensor.shape.as_list()
        
        # 检查维度数量是否一致
        if len(actual_shape) != len(expected_shape):
            raise ValueError(f"Shape mismatch for {name}: expected {expected_shape}, got {actual_shape}")
        
        # 逐个维度检查
        for i, (actual_dim, expected_dim) in enumerate(zip(actual_shape, expected_shape)):
            # 如果 expected_dim 不是 None，且 actual_dim 不是 None，则检查是否匹配
            if expected_dim is not None and actual_dim is not None and actual_dim != expected_dim:
                raise ValueError(f"Shape mismatch for {name}: expected {expected_shape}, got {actual_shape}")
        
        return True

class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention layer with scaled dot-product attention.
    
    Args:
        num_heads: Number of attention heads
        dim: Model dimension (must be divisible by num_heads)
        
    Call args:
        inputs: Input tensor of shape (batch_size, seq_len, dim)
        mask: Optional boolean mask tensor of shape (batch_size, seq_len)
    """
    
    def __init__(self, num_heads, dim):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"Model dimension ({dim}) must be divisible by number of heads ({num_heads})")
            
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.shape_checker = ShapeChecker()

    def build(self, input_shape):
        super().build(input_shape)
        # Projection matrices for Q/K/V
        self.query = keras.layers.Dense(self.dim, use_bias=False)
        self.key = keras.layers.Dense(self.dim, use_bias=False)
        self.value = keras.layers.Dense(self.dim, use_bias=False)
        # Final output projection
        self.output_proj = keras.layers.Dense(self.dim, use_bias=False)
        self.add = keras.layers.Add()
        self.layer_norm = keras.layers.LayerNormalization()
        
    def split_heads(self, x):
        """Split input tensor into multiple heads."""
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.head_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Project inputs to query/key/value
        q = self.query(inputs)  # (B, T, D)
        k = self.key(inputs)    # (B, T, D)
        v = self.value(inputs)  # (B, T, D)
        
        # Split into multiple heads
        q = self.split_heads(q)  # (B, H, T, d)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        # Scaled dot-product attention
        attn_logits = tf.matmul(q, k, transpose_b=True) * self.scale
        self.shape_checker(attn_logits, [batch_size, self.num_heads, seq_len, seq_len])
        
        # Apply mask (if provided)
        if mask is not None:
            mask = mask[:, None, None, :]  # Add head and seq_len dimensions
            attn_logits = tf.where(mask, attn_logits, -1e9)
            
        attn_weights = tf.nn.softmax(attn_logits, axis=-1)
        attn_output = tf.matmul(attn_weights, v)  # (B, H, T, d)
        
        # Merge heads and project
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])  # (B, T, H, d)
        merged = tf.reshape(attn_output, (batch_size, seq_len, self.dim))  # (B, T, D)
        output = self.output_proj(merged)  # (B, T, D)

        x = self.add([x, output])
        output = self.layer_norm(x)
        return output


class FeedForwardLayer(keras.layers.Layer):
    def __init__(self, dim, inner_dim, dropput = 0.1):
        super().__init__()
        self._dim = dim
        self._inner_dim = inner_dim
        self._dropout_r = dropput
    
    def build(self, input_shape):
        super().build(input_shape)
        self.layer1 = keras.layers.Dense(self._inner_dim, activation=keras.activations.relu)
        self.layer2 = keras.layers.Dense(self._dim)
        self.dropout = keras.layers.Dropout(self._dropout_r)
        self.add = keras.layers.Add()
        self.layer_norm = keras.layers.LayerNormalization()
    
    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.dropout(self.layer2(x))
        x = self.add([x, inputs])
        return self.layer_norm(x)
    

class TransformerBlock(keras.layers.Layer):
    def __init__(self, d_model, heads_num, dff):
        super().__init__()
        self._d_model = d_model
        self._heads_num = heads_num
        self._dff = dff

    def build(self, input_shape):
        super().build(input_shape)
        self._mha = MultiHeadAttention(self._heads_num, self._d_model)
        self._ffn = FeedForwardLayer(self._d_model, 4 * self._d_model)
    
    def call(self, inputs, mask=None):
        x = self._mha(inputs, mask)
        x = self._ffn(inputs)
        return x

    
class EmbeddingLayer(keras.layers.Layer):
    def __init__(self, d_model, max_len, vocab_size):
        super().__init__()
        self._d_model = d_model
        self._max_len = max_len
        self._vocab_size = vocab_size

    def build(self, input_shape):
        super().build(input_shape)
        pos = np.arange(self._max_len)[:, np.newaxis]  # [T, 1]
        depth = np.arange(self._d_model / 2)[np.newaxis, :] / (self._d_model / 2)  # [1, d_model / 2]
        angle_rates = 1 / (10000 ** depth)
        inner = pos * angle_rates   # [T, d_model / 2]
        sin_seq = np.sin(inner)
        cos_seq = np.cos(inner)
        self.pos_embedding = np.stack((sin_seq, cos_seq), axis=2).reshape((self._max_len, -1))
        self.pos_embedding = self.pos_embedding[np.newaxis, self._max_len, -1]
        self.embedding_layer = keras.layers.Embedding(self._vocab_size, self._d_model, mask_zero=True)
    
    def call(self, inputs):
        x = self.embedding_layer(inputs)
        final = x + self.pos_embedding
        return final
        