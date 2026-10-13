# tests/selfhost.t — verify self-host full: lexer + parser + evaluator (.t thuần)
# Chạy: python t.py tests/selfhost.t  (kỳ vọng: SELFHOST OK)
import eval
import host

# 1. lexer
toks = host.scan("let x = 1 + 2")
print(toks.len)

# 2. parser
st = [toks, 0]
ast = host.parse_prog(st)
print(ast[0])

# 3. evaluator: số học + fib + tensor + dict + methods + try + import
src = "fn fib(n):\n    if n < 2:\n        return n\n    else:\n        return fib(n - 1) + fib(n - 2)\nprint(fib(10))\nprint(2 ** 10)\nt = tensor([1, 2, 3])\nprint(t + t)\nprint(t @ t)\nd = {\"a\": 1}\nprint(d.get(\"zz\", -1))\nprint(\"hi\".upper())\ntry:\n    raise \"boom-xyz\"\nexcept boom:\n    print(\"caught:\" + err)\nimport ai\nprint(ai.dot(tensor([1, 2]), tensor([3, 4])))\n"
r = eval.run_src(src)
print(r[0])
print("SELFHOST OK")
