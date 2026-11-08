# std/net.t — Native TCP/UDP Sockets & Embedded HTTP Server cho Ngôn ngữ T (v6.0 N4)
let _sys = load("__sys")

struct TcpStream:
    fd

    fn send(self, data):
        return _sys.sock_send(self.fd, str(data))

    fn recv(self, max_bytes = 4096):
        return _sys.sock_recv(self.fd, max_bytes)

    fn close(self):
        return _sys.sock_close(self.fd)

    fn __enter__(self):
        return self

    fn __exit__(self):
        return _sys.sock_close(self.fd)

struct TcpListener:
    fd
    host
    port

    fn accept(self):
        cfd = _sys.sock_accept(self.fd)
        return TcpStream(cfd)

    fn close(self):
        return _sys.sock_close(self.fd)

    fn __enter__(self):
        return self

    fn __exit__(self):
        return _sys.sock_close(self.fd)

struct UdpSocket:
    fd
    port

    fn send_to(self, a, b, c):
        if type(c) == "int":
            return _sys.sock_udp_sendto(self.fd, str(a), str(b), c)
        return _sys.sock_udp_sendto(self.fd, str(c), str(a), b)

    fn recv_from(self, max_bytes = 4096):
        return _sys.sock_udp_recvfrom(self.fd, max_bytes)

    fn close(self):
        return _sys.sock_close(self.fd)

    fn __enter__(self):
        return self

    fn __exit__(self):
        return _sys.sock_close(self.fd)

fn tcp_listen(host = "127.0.0.1", port = 0, backlog = 16):
    fd = _sys.sock_listen(host, port, backlog)
    actual_port = _sys.sock_port(fd)
    return TcpListener(fd, host, actual_port)

fn tcp_connect(host, port):
    fd = _sys.sock_connect(host, port)
    return TcpStream(fd)

fn udp_bind(host = "127.0.0.1", port = 0):
    fd = _sys.sock_udp_open(host, port)
    actual_port = _sys.sock_port(fd)
    return UdpSocket(fd, actual_port)

fn parse_http_request(raw):
    idx = raw.find("\r\n\r\n")
    mut head = raw
    mut body = ""
    if idx >= 0:
        head = raw[0:idx]
        body = raw[idx + 4:raw.len]
    lines = head.split("\r\n")
    mut method = "GET"
    mut path = "/"
    if lines.len > 0:
        parts = lines[0].split(" ")
        if parts.len >= 2:
            method = parts[0]
            path = parts[1]
    mut headers = {}
    mut i = 1
    while i < lines.len:
        ln = lines[i]
        c = ln.find(":")
        if c > 0:
            k = ln[0:c].strip().lower()
            v = ln[c + 1:ln.len].strip()
            headers[k] = v
        i = i + 1
    return {"method": method, "path": path, "headers": headers, "body": body}

fn format_http_response(status, body, content_type = "text/plain; charset=utf-8"):
    status_text = "OK" if status == 200 else ("Not Found" if status == 404 else "Error")
    b = str(body)
    return "HTTP/1.1 " + str(status) + " " + status_text + "\r\nContent-Type: " + content_type + "\r\nContent-Length: " + str(b.len) + "\r\nConnection: close\r\n\r\n" + b

struct HttpServer:
    listener

    fn port(self):
        return self.listener.port

    fn serve_once(self, handler):
        client = self.listener.accept()
        raw = client.recv(8192)
        req = parse_http_request(raw)
        resp = handler(req)
        mut status = 200
        mut body = ""
        mut ctype = "text/plain; charset=utf-8"
        if type(resp) == "dict":
            status = resp["status"] if resp.has("status") else 200
            body = resp["body"] if resp.has("body") else ""
            ctype = resp["content_type"] if resp.has("content_type") else ctype
        else:
            body = str(resp)
        wire = format_http_response(status, body, ctype)
        client.send(wire)
        client.close()
        return req

    fn close(self):
        return self.listener.close()

fn http_server(host = "127.0.0.1", port = 0):
    return HttpServer(tcp_listen(host, port))
