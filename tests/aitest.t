# tests/aitest.t — Kiểm thử chuẩn hóa cho std/ai.t (zeros/ones/dot/relu/mse/grad/linear/softmax)
import ai

# 1. Khởi tạo tensor đặc biệt: zeros & ones
zeros_vec = ai.zeros(3)
ones_vec = ai.ones(3)
assert zeros_vec == tensor([0, 0, 0]), "zeros_vec phải toàn 0"
assert ones_vec == tensor([1, 1, 1]), "ones_vec phải toàn 1"
print("zeros", zeros_vec, "ones", ones_vec)

# 2. Tích vô hướng (dot product)
vec_a = tensor([1, 2, 3])
vec_b = tensor([4, 5, 6])
dot_res = ai.dot(vec_a, vec_b)
assert dot_res == 32, "dot product của [1,2,3] và [4,5,6] phải bằng 32"
print("dot", dot_res)

# 3. Hàm kích hoạt ReLU
relu_input = tensor([-1.0, 0.0, 2.0, -3.5])
relu_res = ai.relu(relu_input)
assert relu_res == tensor([0, 0, 2, 0]), "relu phải triệt tiêu phần âm"
print("relu", relu_res)

# 4. Hàm mất mát Mean Squared Error (MSE)
y_pred = tensor([1.0, 2.0, 3.0])
y_true = tensor([1.5, 2.5, 3.5])
loss = ai.mse(y_pred, y_true)
assert loss == 0.25, "mse của sai lệch 0.5 phải bằng 0.25"
print("mse", loss)

# 5. Vi phân số học (sai phân trung tâm)
# f(t) = t^2 + 3t => f'(t) = 2t + 3 => f'(2) = 7
fn quadratic_func(t):
    return t[0] * t[0] + 3 * t[0]

gradient = ai.grad(quadratic_func, tensor([2.0]))
assert abs(gradient - 7.0) < 0.0001, "f'(2) phải xấp xỉ 7"
print("grad", gradient)

# 6. Biến đổi tuyến tính: weights @ inputs + bias
weights = tensor([[1, 2], [3, 4]])
inputs = tensor([5, 6])
bias = tensor([10, 20])
linear_out = ai.linear(weights, inputs, bias)
assert linear_out == tensor([27, 59]), "kết quả linear phải là [27, 59]"
print("linear", linear_out)

# 7. Hàm Softmax phân phối xác suất
logits = tensor([1.0, 2.0, 3.0])
probs = ai.softmax(logits)
prob_sum = probs.sum()
assert abs(prob_sum - 1.0) < 0.000001, "tổng xác suất softmax phải bằng 1"
print("softmax", probs, "sum", round(prob_sum))

print("AI OK")
