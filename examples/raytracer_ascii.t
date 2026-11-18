# examples/raytracer_ascii.t — 3D Raytracer rendering an ASCII sphere
import math

struct Vec3:
    x
    y
    z
    fn dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z
    fn add(self, o):
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)
    fn sub(self, o):
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)
    fn scale(self, s):
        return Vec3(self.x * s, self.y * s, self.z * s)
    fn norm(self):
        return math.sqrt(self.dot(self))
    fn normalize(self):
        d = self.norm()
        return self.scale(1.0 / d) if d > 1e-9 else self

let shades = " .:-=+*#%@"
let light_dir = Vec3(0.577, 0.577, -0.577).normalize()
let sphere_center = Vec3(0.0, 0.0, 3.0)
let radius = 1.0

let width = 60
let height = 30
let aspect = 2.0  # Terminal character row/column aspect ratio

mut lines = []
for y in range(height):
    # Normalized screen coordinates [-1, 1]
    ny = (float(y) / float(height - 1)) * 2.0 - 1.0
    mut row = ""
    for x in range(width):
        nx = ((float(x) / float(width - 1)) * 2.0 - 1.0) * aspect
        ray_dir = Vec3(nx, ny, 1.5).normalize()
        # Ray-sphere intersection: |origin + t*dir - center|^2 = r^2
        # origin = (0, 0, 0) -> oc = -center
        oc = Vec3(0.0 - sphere_center.x, 0.0 - sphere_center.y, 0.0 - sphere_center.z)
        b = 2.0 * ray_dir.dot(oc)
        c = oc.dot(oc) - radius * radius
        disc = b * b - 4.0 * c
        if disc > 0.0:
            t = (0.0 - b - math.sqrt(disc)) * 0.5
            hit = ray_dir.scale(t)
            normal = hit.sub(sphere_center).normalize()
            diffuse = normal.dot(light_dir)
            if diffuse < 0.0:
                diffuse = 0.0
            idx = int(diffuse * float(shades.len - 1))
            if idx >= shades.len:
                idx = shades.len - 1
            row = row + shades[idx:idx + 1]
        else:
            row = row + " "
    lines.append(row)

print("=== 3D Raytraced ASCII Sphere ===")
print("\n".join(lines))
