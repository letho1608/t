# std/os.t — giao diện hệ điều hành, tệp tin, tiến trình và nén gói của T

let _sys = load("__sys")

fn platform():
    return _sys.platform()

fn exepath():
    return _sys.exepath()

fn getcwd():
    return _sys.getcwd()

fn chdir(p):
    return _sys.chdir(p)

fn abspath(p):
    return _sys.abspath(p)

fn getenv(k, dflt = ""):
    v = _sys.getenv(k)
    return dflt if v == "" and dflt != "" else v

fn setenv(k, v):
    return _sys.setenv(k, v)

fn exists(p):
    return _sys.exists(p)

fn isfile(p):
    return _sys.isfile(p)

fn isdir(p):
    return _sys.isdir(p)

fn mtime(p):
    return _sys.mtime(p)

fn mkdir(p):
    return _sys.mkdir(p)

fn makedirs(p):
    return _sys.mkdir(p)

fn remove(p):
    return _sys.remove(p)

fn rmtree(p):
    return _sys.remove(p)

fn listdir(p = "."):
    return _sys.listdir(p)

fn mktemp(prefix = "ttmp"):
    return _sys.mktemp(prefix)

fn sha256(s):
    return _sys.sha256(s)

fn read_file(p):
    f = open(p, "r")
    s = read(f)
    close(f)
    return s

fn write_file(p, content):
    f = open(p, "w")
    n = write(f, content)
    close(f)
    return n

fn copy_file(src, dst):
    data = read_file(src)
    return write_file(dst, data)

fn sha256_file(p):
    return _sys.sha256(read_file(p))

fn run(args_list, input_str = "", cwd_str = nil, env_dict = nil):
    if env_dict != nil:
        return _sys.run(args_list, input_str, cwd_str if cwd_str != nil else "", env_dict)
    if cwd_str != nil:
        return _sys.run(args_list, input_str, cwd_str)
    if input_str != "":
        return _sys.run(args_list, input_str)
    return _sys.run(args_list)

fn stdin_line():
    return _sys.stdin_line()

fn stdin_read(n):
    return _sys.stdin_read(n)

fn stdout_write(s):
    return _sys.stdout_write(s)

fn stderr_write(s):
    return _sys.stderr_write(s)

fn zip_pack(entries, out_path):
    return _sys.zip_pack(entries, out_path)

fn zip_unpack(zip_path, dest_dir):
    return _sys.zip_unpack(zip_path, dest_dir)

fn exit(code = 0):
    return _sys.exit(code)
