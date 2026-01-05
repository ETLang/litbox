using System;
using System.Runtime.InteropServices;
using UnityEngine;

public static class ResourceExtensions
{
    public static RenderTexture CreateRWTexture(this DisposalHelperComponent _this, int w, int h, RenderTextureFormat format) {
        var output = new RenderTexture(w, h, 0, format)
        {
            enableRandomWrite = true,
            useMipMap = false,
        };
        output.Create();
        _this.DisposeOnDisable(() => GameObject.DestroyImmediate(output));
        return output;
    }

    public static RenderTexture CreateRWTextureWithMips(this DisposalHelperComponent _this, int w, int h, RenderTextureFormat format, int d = 0) {
        var output = new RenderTexture(w, h, d, format)
        {
            enableRandomWrite = true,
            useMipMap = true,
            autoGenerateMips = false,
        };
        output.Create();
        output.GenerateMips();
        _this.DisposeOnDisable(() => GameObject.DestroyImmediate(output));
        return output;
    }

    public static ComputeBuffer CreateStructuredBuffer<T>(this DisposalHelperComponent _this, T[] data) {
        var structureSize = Marshal.SizeOf<T>();
        if(structureSize % 16 != 0) {
            throw new Exception($"{typeof(T).Name}'s size is {structureSize}. It must be a multiple of 16");
        }

        var output = new ComputeBuffer(data.Length, Marshal.SizeOf<T>(), ComputeBufferType.Structured, ComputeBufferMode.Immutable);
        output.SetData(data);
        _this.DisposeOnDisable(output);
        return output;
    }
}