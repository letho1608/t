# bench/ai_vs_ag.t — So sánh hiệu năng giữa inference thuần (std/ai.t) và autograd (std/autograd.t)
import ai
import autograd as ag

# Số lần lặp forward inference cho mỗi mẫu
FORWARD_PASSES = 10

# Tốc độ học cho bộ tối ưu SGD trong huấn luyện autograd
LEARNING_RATE = 0.01

# Đo hiệu năng suy luận (forward pass) dùng thư viện ai: ReLU và Softmax
fn bench_ai(sample_count):
    features = tensor([i * 0.01 - 0.5 for i in range(sample_count)])
    mut accumulated_sum = 0.0
    for step in range(FORWARD_PASSES):
        activated = ai.relu(features)
        accumulated_sum = accumulated_sum + activated.sum()
        probabilities = ai.softmax(features)
        accumulated_sum = accumulated_sum + probabilities[0]
    return accumulated_sum

# Đo hiệu năng huấn luyện mô hình MLP (forward + backward + sgd) dùng autograd
fn bench_ag(steps):
    seed(1)
    weights = ag.param(randn([4, 4]))
    inputs = ag.const(randn([1, 4]))
    mut last_loss = 0.0
    for step in range(steps):
        hidden_layer = ag.tanh(ag.matmul(inputs, weights))
        loss = ag.mean(hidden_layer)
        ag.backward(loss)
        ag.sgd([weights], LEARNING_RATE)
        last_loss = loss.val.item()
    return last_loss

print("ai", bench_ai(5000))
print("ag", bench_ag(500))
