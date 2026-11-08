# std/ai.t — Thư viện tiện ích AI và học máy cho T
# Cung cấp các hàm toán học, activation, loss function và vi phân số học.

# Bước sai phân mặc định cho đạo hàm số học
DIFF_STEP = 0.00001

# Tạo tensor chứa toàn số 0 theo kích thước hoặc shape truyền vào
fn zeros(shape):
    return full(shape, 0)

# Tạo tensor chứa toàn số 1 theo kích thước hoặc shape truyền vào
fn ones(shape):
    return full(shape, 1)

# Tích vô hướng (dot product) hoặc nhân ma trận giữa hai tensor
fn dot(a, b):
    return a @ b

# Hàm kích hoạt ReLU (Rectified Linear Unit): max(0, x)
fn relu(x):
    t = x if type(x) == "tensor" else tensor(x)
    return t.relu()

# Hàm mất mát Mean Squared Error (MSE): trung bình bình phương sai số
fn mse(predicted, actual):
    diff = predicted - actual
    return (diff * diff).mean()

# Tính đạo hàm số học f'(x) tại điểm x bằng sai phân trung tâm (central difference)
fn grad(f, x):
    h = DIFF_STEP
    return (f(x + h) - f(x - h)) / (2 * h)

# Biến đổi tuyến tính affine: weights @ inputs + bias
fn linear(weights, inputs, bias):
    return (weights @ inputs) + bias

# Hàm Softmax chuẩn hóa phân phối xác suất
fn softmax(x):
    t = x if type(x) == "tensor" else tensor(x)
    return t.softmax()

# Hàm kích hoạt GELU chính xác (qua erf) — forward gộp trong kernel C
fn gelu(x):
    t = x if type(x) == "tensor" else tensor(x)
    return t.gelu()

# Tích chập 2D NCHW (hoặc 2D [H, W]): forward gộp trong kernel C
fn conv2d(x, w, bias = nil, stride = 1, pad = 0):
    tx = x if type(x) == "tensor" else tensor(x)
    tw = w if type(w) == "tensor" else tensor(w)
    tb = (bias if type(bias) == "tensor" else tensor(bias)) if bias != nil else nil
    return tx.conv2d(tw, tb, stride, pad)
