# examples/neural_xor.t — Huấn luyện mạng MLP 2 lớp giải bài toán XOR bằng autograd & sgd
import autograd as ag

print("=== Huấn luyện Mạng Nơ-ron giải bài toán phi tuyến XOR ===")
seed(1)

let X = ag.const(tensor([[0, 0], [0, 1], [1, 0], [1, 1]]))
let Y = tensor([[0], [1], [1], [0]])

mut W1 = ag.param(randn([2, 4]))
mut b1 = ag.param(zeros([1, 4]))
mut W2 = ag.param(randn([4, 1]))
mut b2 = ag.param(zeros([1, 1]))
let params = [W1, b1, W2, b2]

mut loss = nil
for step in range(500):
    h = ag.tanh(ag.add(ag.matmul(X, W1), b1))
    p = ag.sigmoid(ag.add(ag.matmul(h, W2), b2))
    t1 = ag.mul(ag.log(p), Y)
    t2 = ag.mul(ag.log(ag.sub(1, p)), 1 - Y)
    loss = ag.mul(ag.mean(ag.add(t1, t2)), -1)
    ag.backward(loss)
    ag.sgd(params, 1.0)
    if (step + 1) % 100 == 0:
        lval = round(loss.val.item() * 10000.0) / 10000.0
        print(f"Bước {step + 1} / 500 | Loss: {lval}")

print("\n=== Kết quả dự đoán sau huấn luyện ===")
pred = ((X.val @ W1.val + b1.val).tanh() @ W2.val + b2.val).sigmoid()
let inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
let targets = [0, 1, 1, 0]
let probs = pred.flatten().tolist()
for i in range(4):
    in_a = inputs[i][0]
    in_b = inputs[i][1]
    target = targets[i]
    prob = round(probs[i] * 1000.0) / 1000.0
    ans = 1 if prob > 0.5 else 0
    print(f"XOR({in_a}, {in_b}) -> Kỳ vọng: {target} | Dự đoán: {ans} (xác suất: {prob})")
