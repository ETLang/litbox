using System;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

public static class ResourceExtensions
{
    public static RenderTexture CreateRWTexture(this DisposalHelperComponent _this, int w, int h, RenderTextureFormat format) {
        var output = BufferManager.AcquireTexture(w, h, format);
        _this.DisposeOnDisable(() => BufferManager.Release(ref output));
        return output;
    }

    public static RenderTexture CreateRWTexture(this Disposable _this, int w, int h, RenderTextureFormat format) {
        var output = BufferManager.AcquireTexture(w, h, format);
        _this.AutoDispose(() => BufferManager.Release(ref output));
        return output;
    }

    public static RenderTexture CreateRWTextureWithMips(this DisposalHelperComponent _this, int w, int h, RenderTextureFormat format, int d = 0) {
        var output = BufferManager.AcquireTexture3D(w, h, d, format, true);
        _this.DisposeOnDisable(() => BufferManager.Release(ref output));
        return output;
    }

    public static RenderTexture CreateRWTextureWithMips(this Disposable _this, int w, int h, RenderTextureFormat format, int d = 0) {
        var output = BufferManager.AcquireTexture3D(w, h, d, format, true);
        _this.AutoDispose(() => BufferManager.Release(ref output));
        return output;
    }
    
    public static ComputeBuffer CreateStructuredBuffer<T>(this DisposalHelperComponent _this, T[] data) {
        var output = BufferManager.AcquireBuffer(data);
        _this.DisposeOnDisable(() => BufferManager.Release(ref output));
        return output;
    }

    public static ComputeBuffer CreateStructuredBuffer<T>(this Disposable _this, T[] data) {
        var output = BufferManager.AcquireBuffer(data);
        _this.AutoDispose(() => BufferManager.Release(ref output));
        return output;
    }

    public static int MipWidth(this Texture _this, int mipLevel) {
        return Mathf.Max(1, _this.width >> mipLevel);
    }

    public static int MipHeight(this Texture _this, int mipLevel) {
        return Mathf.Max(1, _this.height >> mipLevel);
    }

    public static async Task<NativeArray<T>> ReadbackAsync<T>(this ComputeBuffer _this) where T : struct
    {
        var result = await AsyncGPUReadback.RequestAsync(_this);
        if(result.hasError)
        {
            throw new Exception("GPU Readback Error");
        }
        return result.GetData<T>();
    }
}