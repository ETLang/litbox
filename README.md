# Photoner

A 2D interpretation of ray tracing.

Real time ray tracing in games has a tendency to focus on 3D photorealism. Photoner reimagines ray tracing in the 2D realm. The objective is to enhance 2D games with a look and feel never before seen in the game industry, and empower artists with tools to immerse players in fantastical lighting that can't be found elsewhere, even in 3D.

---

## Features

- Illumination of 2D colloidal materials
- (TODO) Solid object reflection and refraction
- (TODO) Support for multiple interacting illuminated layers
- HDR output with customizable tone mapping

---

## Getting Started

TODO

---

## How it Works

Photoner simulates hundreds of thousands of photons every frame. The result is a grainy, noisy light map. This is image is then fed into a custom denoising AI that transforms it into a crisp, clean lightmap that can be used to illuminate your scene. Multiple layers are combined as necessary, and the HDR composition is then tone mapped for a pristine final output image. Photoner manages lightmap layers, and provides tone mapping, while leaving the overall scene composition to the supporting engine.

---

## Techniques

Photoner deliberately avoids using pre-packaged ray tracing or denoising solutions like RTX or DLSS, though in the future leveraging such technologies may be worthwhile. The current implementation demonstrates:

- Highly parallelized GPU-based ray tracing
- Importance sampling
- Hybrid (forward + backward) path tracing
- UNet-based denoising and upsampling
- (TODO more to come..)
