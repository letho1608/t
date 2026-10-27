# std/safetensors.t — Parser và Reader cho định dạng mô hình Hugging Face Safetensors

import mmap
import json

struct SafetensorsReader:
    mm
    header
    header_size
    data_start

    fn get_tensor(self, name):
        if not self.header.has(name):
            return nil
        let info = self.header[name]
        let dtype = info["dtype"]
        let shape = info["shape"]
        let offsets = info["data_offsets"]
        let begin = offsets[0]
        let file_offset = self.data_start + begin

        if dtype == "F32":
            return self.mm.tensor_f32(file_offset, shape)
        if dtype == "F16":
            return self.mm.tensor_f16(file_offset, shape)
        return nil

    fn metadata(self):
        return self.header.get("__metadata__", {})

    fn tensor_names(self):
        mut names = []
        for k in self.header:
            if k != "__metadata__":
                names.append(k)
        return names

    fn close(self):
        self.mm.close()

fn open_safetensors(path):
    let mm = mmap.open(path)
    let header_size = mm.read_u64(0)
    let header_json = mm.read_bytes(8, header_size)
    let header = json.loads(header_json)
    let data_start = 8 + header_size
    return SafetensorsReader(mm, header, header_size, data_start)
