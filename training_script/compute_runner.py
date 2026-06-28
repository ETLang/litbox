import torch
import compushady
from compushady import Texture2D, Buffer, HEAP_UPLOAD, HEAP_READBACK
from compushady.formats import R32_FLOAT, R32G32_FLOAT, R32G32B32A32_FLOAT
from compushady.shaders import hlsl
import numpy as np
import os
from io import StringIO
from pcpp import Preprocessor
import data_processing

# Compushady did not implement a format propery accessor,
# so this tacks it on manually.
def make_texture(width, height, format):
    texture = Texture2D(width, height, format)
    texture.format = format
    return texture

class ComputeShaderRunner:
    def __init__(self):
        # self.i = 0

        old_dir = os.getcwd()
        compushady.config.set_debug(True)
        pp = Preprocessor()
        pp.add_path('./Assets/Resources')
        pp.add_path('./UnityIncludes')
        pp.define('sampler2D sampler')
        pp.define('samplerCUBE sampler')
        pp.line_directive = None
        with open('./Assets/Resources/Variance.compute', 'r') as f:
            variance_code = f.read()
        pp.parse(variance_code)
        processed_code = StringIO()
        pp.write(processed_code)
        processed_code = processed_code.getvalue()
        self.cs_variance = hlsl.compile(processed_code, 'Variance_Kernel')

        # shader = """
        #     RWTexture2D<uint4> texture : register(u0);
        #     [numthreads(2, 2, 1)]
        #     void main(int3 tid : SV_DispatchThreadID)
        #     {
        #         uint4 color = texture[tid.xy];
        #         uint red = color.r;
        #         color.r = color.g;
        #         color.g = red;
        #         texture[tid.xy] = color;
        #     }
        #     """
        # target_texture = Texture2D(256, 256, R32G32B32A32_FLOAT)
        # compute = compushady.Compute(hlsl.compile(shader), uav=[target_texture])

    def tensor_to_texture(self, tensor, batch_n):
        _, _, height, width = tensor.shape

        # Compushady expects (Height, Width, Channels) or flattened raw data
        selected_batch = tensor[batch_n].permute(1, 2, 0).contiguous()

        # Convert RGB tensors to RGBA
        if selected_batch.shape[2] == 3:
            alpha_channel = torch.zeros(height, width, 1, dtype=selected_batch.dtype, device=selected_batch.device)
            selected_batch = torch.cat([selected_batch, alpha_channel], dim=2)

        # Only RGBAFloat is supported for now.
        target_texture = make_texture(width, height, R32G32B32A32_FLOAT)

        # Size = Width * Height * 4 channels * 4 bytes (float32)
        data_size = width * height * 4 * 4
        staging_buffer = Buffer(data_size, HEAP_UPLOAD)

        raw_data = selected_batch.cpu().numpy().tobytes()
        staging_buffer.upload(raw_data)
        staging_buffer.copy_to(target_texture)
        return target_texture

    def texture_to_tensor(self, texture):
        width = texture.width
        height = texture.height
        format = texture.format
        
        # Determine number of channels based on format
        if format == R32_FLOAT:
            channels = 1
        elif format == R32G32B32A32_FLOAT:
            channels = 4
        elif format == R32G32_FLOAT:
            channels = 2
        else:
            raise ValueError("Unsupported texture format")
        
        # Size = Width * Height * Channels * 4 bytes (float32)
        data_size = width * height * channels * 4
        staging_buffer = Buffer(data_size, HEAP_READBACK)
        
        texture.copy_to(staging_buffer)
        raw_data = staging_buffer.readback(data_size)
        numpy_array = np.frombuffer(raw_data, dtype=np.float32).reshape(height, width, channels)
        return torch.from_numpy(numpy_array.copy()).permute(2, 0, 1).float()

    def compute_variance(self, input_a, input_b, albedo):
        batch_size, channels, height, width = input_a.shape
        
        if channels != 3 and channels != 4:
            raise ValueError("Variance computation only operates on RGB or RGBA tensors")

        results = []
        output_texture = make_texture(width, height, R32_FLOAT)
        for batch_idx in range(batch_size):
            input_a_texture = self.tensor_to_texture(input_a, batch_idx)
            input_b_texture = self.tensor_to_texture(input_b, batch_idx)
            albedo_texture = self.tensor_to_texture(albedo, batch_idx)

            # Important to get the bindings exactly right. Errors here are not informative.
            pipeline = compushady.Compute(self.cs_variance, srv=[input_a_texture, input_b_texture, albedo_texture], uav=[output_texture])
            pipeline.dispatch((width - 1) // 16 + 1, (height - 1) // 16 + 1, 1)
            t = self.texture_to_tensor(output_texture)
            # data_processing.save_exr(f"./Test_{self.i}.exr", t)
            # self.i = self.i + 1
            results.append(t)
        
        return torch.stack(results, dim=0)