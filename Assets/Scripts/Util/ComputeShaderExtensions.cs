using System;
using Unity.Mathematics;
using UnityEngine;

public static class ComputeShaderExtensions
{
    public static int3 GetThreadGroupCount(this ComputeShader shader, int kernelID, int threadsX, int threadsY, int threadsZ)
    {
        shader.GetKernelThreadGroupSizes(kernelID, out var groupSizeX, out var groupSizeY, out var groupSizeZ);
        return new int3(
            (int)((threadsX - 1) / groupSizeX + 1),
            (int)((threadsY - 1) / groupSizeY + 1),
            (int)((threadsZ - 1) / groupSizeZ + 1));
    }

    public static void RunKernel(this ComputeShader shader, string kernel, int n, params (string,object)[] args) {
        RunKernel(shader, kernel, n, 1, args);
    }

    public static void RunKernel(this ComputeShader shader, int kernelID, int n, params (string,object)[] args) {
        RunKernel(shader, kernelID, n, 1, args);
    }

    public static void RunKernel(this ComputeShader shader, string kernel, int w, int h, params (string,object)[] args)
    {
        RunKernel(shader, shader.FindKernel(kernel), w, h, args);
    }

    public static void RunKernel(this ComputeShader shader, int kernelID, int w, int h, params (string,object)[] args) {
        foreach(var tuple in args) {
            switch(tuple.Item2) {
            case null:
                break;
            case Texture texture:
                shader.SetTexture(kernelID, tuple.Item1, texture);
                shader.SetVector($"lut_window_{tuple.Item1}", new Vector4(
                    0.5f / texture.width, 1 - 1.0f / texture.width,
                    0.5f / texture.height, 1 - 1.0f / texture.height));

                if(texture is Texture3D tex3D) {
                    shader.SetVector($"lut_slice_window_{tuple.Item1}", new Vector2(
                        0.5f / tex3D.depth, 1 - 1.0f / tex3D.depth));
                }
                break;
            case ComputeBuffer buffer:
                shader.SetBuffer(kernelID, tuple.Item1, buffer);
                break;
            case MipSpec mipSpec:
                shader.SetTexture(kernelID, tuple.Item1, mipSpec.texture, mipSpec.mipLevel);
                break;
            case int ix:
                shader.SetInt(tuple.Item1, ix);
                break;
            case float fx:
                shader.SetFloat(tuple.Item1, fx);
                break;
            case Vector2 v2:
                shader.SetVector(tuple.Item1, v2);
                break;
            case Vector3 v3:
                shader.SetVector(tuple.Item1, v3);
                break;
            case Vector4 v4:
                shader.SetVector(tuple.Item1, v4);
                break;
            default:
                throw new Exception("What is " + tuple.Item2.GetType().Name + "?");
            }
        }

        shader.DispatchAutoGroup(kernelID, w, h, 1);
    }

    public static void DispatchAutoGroup(this ComputeShader shader, int kernelID, int threadsX, int threadsY, int threadsZ)
    {
        var groupCount = shader.GetThreadGroupCount(kernelID, threadsX, threadsY, threadsZ);
        shader.Dispatch(kernelID, groupCount.x, groupCount.y, groupCount.z); 
    }

    public static void SetShaderFlag(this ComputeShader shader, string keyword, bool value) {
        var id = shader.keywordSpace.FindKeyword(keyword);
        shader.SetKeyword(id, value);
    }

    public class MipSpec {
        public Texture texture;
        public int mipLevel;
    }

    public static MipSpec SelectMip(this Texture texture, int mip)
    {
        return new MipSpec {
            texture = texture,
            mipLevel = mip,
        };
    }
}