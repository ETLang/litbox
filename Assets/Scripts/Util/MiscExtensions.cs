using System;
using System.Collections;
using System.Linq;
using UnityEngine;
using UnityEngine.UIElements;
using Unity.Mathematics;
using UnityEngine.Experimental.Rendering;

public static class TextureExtensions {
    public static T ReadClamped<T>(this T[,] arr, int i, int j) {
        return arr[
            i < 0 ? 0 : (i >= arr.GetLength(0) ? arr.GetLength(0) - 1 : i),
            j < 0 ? 0 : (j >= arr.GetLength(1) ? arr.GetLength(1) - 1 : j)
        ];
    }

    public static T ReadBounded<T>(this T[,] arr, int i, int j, T bound) {
        if(i < 0 || j < 0 || i >= arr.GetLength(0) || j >= arr.GetLength(1)) {
            return bound;
        }
        return arr[i,j];
    }

    public static float ReadClampScaled(this float[,] arr, int i, int j, float clampScale) {
        float scale = 1;
        if(i < 0 || j < 0 || i >= arr.GetLength(0) || j >= arr.GetLength(1)) {
            scale = clampScale;
        }
        return arr.ReadClamped(i,j) * scale;
    }

    public static T ReadMirrored<T>(this T[,] arr, int i, int j) {
        var w = arr.GetLength(0) - 1;
        var h = arr.GetLength(1) - 1;

        var sw = 2*w;
        var sh = 2*h;
        
        var p_i = i % sw;
        var p_j = j % sh;

        if(p_i < 0) {
            p_i += sw;
        }

        if(p_j < 0) {
            p_j += sh;
        }

        return arr[Math.Min(p_i,sw-p_i), Math.Min(p_j,sh-p_j)];
    }

    public static float ReadCubic(this float[] arr, float index) {
        float p0,p1,p2,p3;

        if(index < 0) throw new IndexOutOfRangeException();
        if(index > arr.Length - 1)
            throw new IndexOutOfRangeException();

        if(arr.Length == 1) return arr[0];
        if(arr.Length == 2) return arr[0] + index * (arr[1] - arr[0]);

        if(index < 1) {
            p1 = arr[0];
            p2 = arr[1];
            p3 = arr[2];
            p0 = 3*p1 - 3*p2 + p3;
        } else if(index >= arr.Length - 2) {
            p0 = arr[arr.Length - 3];
            p1 = arr[arr.Length - 2];
            p2 = arr[arr.Length - 1];
            p3 = p0 - 3*p1 + 3*p2;
        } else {
            var i = (int)index;
            p0 = arr[i-1];
            p1 = arr[i];
            p2 = arr[i+1];
            p3 = arr[i+2];
        }

        float x = index - (int)index;
        float xx = x*x;
        float xxx = xx*x;

        return (-0.5f*p0 + 1.5f*p1 - 1.5f*p2 + 0.5f*p3)*xxx + (p0 - 2.5f*p1  + 2.0f*p2 - 0.5f*p3)*xx + (-0.5f*p0 + 0.5f*p2)*x + p1;
    }

    public static System.Collections.Generic.IEnumerable<T> Flat<T>(this T[,] arr) {
        var i = arr.GetEnumerator();
        while(i.MoveNext()) {
            yield return (T)i.Current;
        }
    }

    public static Texture AsTexture(this float[] lut) {
        var texture = new Texture2D(lut.Length, 1, TextureFormat.RFloat, false, true);
        texture.SetPixels(lut.Select(x => new Color(x, 0, 0, 0)).ToArray());
        texture.Apply();
        return texture;
    }

    public static Texture AsTexture(this float2[] lut) {
        var texture = new Texture2D(lut.Length, 1, TextureFormat.RGFloat, false, true);
        texture.SetPixels(lut.Select(v => new Color(v.x, v.y, 0, 0)).ToArray());
        texture.Apply();
        return texture;
    }

    public static Texture AsTexture(this float3[] lut) {
        var texture = new Texture2D(lut.Length, 1, TextureFormat.RGBAFloat, false, true);
        texture.SetPixels(lut.Select(v => new Color(v.x, v.y, v.z, 0)).ToArray());
        texture.Apply();
        return texture;
    }

    public static Texture AsTexture(this float4[] lut) {
        var texture = new Texture2D(lut.Length, 1, TextureFormat.RGFloat, false, true);
        texture.SetPixels(lut.Select(v => new Color(v.x, v.y, v.z, v.w)).ToArray());
        texture.Apply();
        return texture;
    }

    public static Texture AsTexture(this float[,] lut) {
        var texture = new Texture2D(lut.GetLength(0), lut.GetLength(1), TextureFormat.RFloat, false, true);
        Color[] c = new Color[lut.GetLength(0) * lut.GetLength(1)];
        for(int i = 0;i < lut.GetLength(0);i++) {
            for(int j = 0;j < lut.GetLength(1);j++) {
                var val = lut[i,j];
                c[i + j * lut.GetLength(0)] = new Color(val, 0, 0, 0);
            }
        }
        texture.SetPixels(c);
        texture.Apply();
        return texture;
    }

    public static Texture AsTexture(this float2[,] lut) {
        var texture = new Texture2D(lut.GetLength(0), lut.GetLength(1), TextureFormat.RGFloat, false, true);
        Color[] c = new Color[lut.GetLength(0) * lut.GetLength(1)];
        for(int i = 0;i < lut.GetLength(0);i++) {
            for(int j = 0;j < lut.GetLength(1);j++) {
                var val = lut[i,j];
                c[i + j * lut.GetLength(0)] = new Color(val.x, val.y, 0, 0);
            }
        }
        texture.SetPixels(c);
        texture.Apply();
        return texture;
    }

    public static Texture AsTexture(this float3[,] lut) {
        var texture = new Texture2D(lut.GetLength(0), lut.GetLength(1), TextureFormat.RGBAFloat, false, true);
        Color[] c = new Color[lut.GetLength(0) * lut.GetLength(1)];
        for(int i = 0;i < lut.GetLength(0);i++) {
            for(int j = 0;j < lut.GetLength(1);j++) {
                var val = lut[i,j];
                c[i + j * lut.GetLength(0)] = new Color(val.x, val.y, val.z, 0);
            }
        }
        texture.SetPixels(c);
        texture.Apply();
        return texture;
    }

    public static Texture AsTexture(this float4[,] lut) {
        var texture = new Texture2D(lut.GetLength(0), lut.GetLength(1), TextureFormat.RGBAFloat, false, true);
        Color[] c = new Color[lut.GetLength(0) * lut.GetLength(1)];
        for(int i = 0;i < lut.GetLength(0);i++) {
            for(int j = 0;j < lut.GetLength(1);j++) {
                var val = lut[i,j];
                c[i + j * lut.GetLength(0)] = new Color(val.x, val.y, val.z, val.w);
            }
        }
        texture.SetPixels(c);
        texture.Apply();
        return texture;
    }

    public static Texture AsTexture(this float[,,] lut) {
        var texture = new Texture3D(lut.GetLength(0), lut.GetLength(1), lut.GetLength(2), TextureFormat.RFloat, false, true);
        Color[] c = new Color[lut.GetLength(0) * lut.GetLength(1) * lut.GetLength(2)];
        for(int i = 0;i < lut.GetLength(0);i++) {
            for(int j = 0;j < lut.GetLength(1);j++) {
                for(int k = 0;k < lut.GetLength(2);k++) {
                    var val = lut[i,j,k];
                    c[i + j * lut.GetLength(0) + k * lut.GetLength(0) * lut.GetLength(1)] = new Color(val, 0, 0, 0);
                }
            }
        }
        texture.SetPixels(c);
        texture.Apply();
        return texture;
    }

    public static Texture AsTexture(this float2[,,] lut) {
        var texture = new Texture3D(lut.GetLength(0), lut.GetLength(1), lut.GetLength(2), TextureFormat.RGFloat, false, true);
        Color[] c = new Color[lut.GetLength(0) * lut.GetLength(1) * lut.GetLength(2)];
        for(int i = 0;i < lut.GetLength(0);i++) {
            for(int j = 0;j < lut.GetLength(1);j++) {
                for(int k = 0;k < lut.GetLength(2);k++) {
                    var val = lut[i,j,k];
                    c[i + j * lut.GetLength(0) + k * lut.GetLength(0) * lut.GetLength(1)] = new Color(val.x, val.y, 0, 0);
                }
            }
        }
        texture.SetPixels(c);
        texture.Apply();
        return texture;
    }

    public static Texture AsTexture(this float3[,,] lut) {
        var texture = new Texture3D(lut.GetLength(0), lut.GetLength(1), lut.GetLength(2), TextureFormat.RGBAFloat, false, true);
        Color[] c = new Color[lut.GetLength(0) * lut.GetLength(1) * lut.GetLength(2)];
        for(int i = 0;i < lut.GetLength(0);i++) {
            for(int j = 0;j < lut.GetLength(1);j++) {
                for(int k = 0;k < lut.GetLength(2);k++) {
                    var val = lut[i,j,k];
                    c[i + j * lut.GetLength(0) + k * lut.GetLength(0) * lut.GetLength(1)] = new Color(val.x, val.y, val.z, 0);
                }
            }
        }
        texture.SetPixels(c);
        texture.Apply();
        return texture;
    }

    public static Texture AsTexture(this float4[,,] lut) {
        var texture = new Texture3D(lut.GetLength(0), lut.GetLength(1), lut.GetLength(2), TextureFormat.RGBAFloat, false, true);
        Color[] c = new Color[lut.GetLength(0) * lut.GetLength(1) * lut.GetLength(2)];
        for(int i = 0;i < lut.GetLength(0);i++) {
            for(int j = 0;j < lut.GetLength(1);j++) {
                for(int k = 0;k < lut.GetLength(2);k++) {
                    var val = lut[i,j,k];
                    c[i + j * lut.GetLength(0) + k * lut.GetLength(0) * lut.GetLength(1)] = new Color(val.x, val.y, val.z, val.w);
                }
            }
        }
        texture.SetPixels(c);
        texture.Apply();
        return texture;
    }

    // Slow. Do not use frequently.
    public static float4[,] ForceReadData(this Texture tex, int miplevel = 0) {
        var tex2d = tex as Texture2D;
        var tex3d = tex as Texture3D;
        var renderTex = tex as RenderTexture;

        var w = tex.MipWidth(miplevel);
        var h = tex.MipHeight(miplevel);

        Color[] pixels;

        if(renderTex != null) {
            Texture2D readableTex = new Texture2D(w, h, GraphicsFormatUtility.GetTextureFormat(tex.graphicsFormat), false, !tex.isDataSRGB);

            Graphics.SetRenderTarget(renderTex, miplevel);
            readableTex.ReadPixels(new Rect(0, 0, w, h), 0, 0);
            readableTex.Apply();
            pixels = readableTex.GetPixels();
            GameObject.DestroyImmediate(readableTex);
        } else if(tex2d != null) {
            Texture2D readableTex = new Texture2D(w, h, GraphicsFormatUtility.GetTextureFormat(tex.graphicsFormat), false, !tex.isDataSRGB);
            Graphics.CopyTexture(tex2d, 0, miplevel, readableTex, 0, 0);
            pixels = readableTex.GetPixels(miplevel);
        } else if(tex3d != null) {
            throw new NotImplementedException("ForceReadData not implemented for Texture3D.");
        } else {
            throw new NotImplementedException("ForceReadData - Unknown texture type");
        }

        var result = new float4[w, h];

        for(int j = 0;j < h;j++) {
            for(int i = 0;i < w;i++) {
                var c = pixels[i + j * w];
                result[i,j] = new float4(c.r, c.g, c.b, c.a);
            }
        }
        return result;
    }

    public static void SaveTextureEXR(this RenderTexture target, string path)
    {
        if(target.format != RenderTextureFormat.ARGBFloat) {
            var descriptor = target.descriptor;
            descriptor.colorFormat = RenderTextureFormat.ARGBFloat;
            descriptor.sRGB = false;
            descriptor.mipCount = 1;

            var floatTarget = new RenderTexture(descriptor);
            floatTarget.Create();

            var current = RenderTexture.active;
            Graphics.Blit(target, floatTarget);
            RenderTexture.active = current;
            target = floatTarget;
        }

        Texture2D image = new Texture2D(target.width, target.height, TextureFormat.RGBAFloat, false, true);

        Graphics.SetRenderTarget(target);
        image.ReadPixels(new Rect(0, 0, image.width, image.height), 0, 0);
        image.Apply();

        byte[] bytes = image.EncodeToEXR(Texture2D.EXRFlags.CompressZIP);
        System.IO.File.WriteAllBytes(path, bytes);
    }

    public static void SaveTexturePNG(this RenderTexture target, string path)
    {
        if(!target.sRGB) {
            var descriptor = target.descriptor;
            descriptor.colorFormat = RenderTextureFormat.ARGB32;
            descriptor.sRGB = true;
            descriptor.mipCount = 1;

            var srgbTarget = new RenderTexture(descriptor);
            srgbTarget.Create();

            var current = RenderTexture.active;
            Graphics.Blit(target, srgbTarget);
            RenderTexture.active = current;
            target = srgbTarget;
        }

        Texture2D image = new Texture2D(target.width, target.height, TextureFormat.RGBA32, false, true);

        Graphics.SetRenderTarget(target);
        image.ReadPixels(new Rect(0, 0, image.width, image.height), 0, 0);
        image.Apply();

        byte[] bytes = image.EncodeToPNG();
        System.IO.File.WriteAllBytes(path, bytes);

        GameObject.DestroyImmediate(image);
    }

}