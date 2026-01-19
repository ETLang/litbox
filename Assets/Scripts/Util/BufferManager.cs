using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Unity.Mathematics;
using Unity.VisualScripting.Dependencies.Sqlite;
using UnityEngine;

/// <summary>
/// Manages textures and buffers for efficient reuse, and provides global specialized buffers.
/// </summary>
public static class BufferManager
{
    static Dictionary<(int,int,int,RenderTextureFormat,bool), List<RenderTexture>> _textureLibrary = 
        new Dictionary<(int, int, int, RenderTextureFormat, bool), List<RenderTexture>>();
    static Dictionary<(int,int),List<ComputeBuffer>> _bufferLibrary = 
        new Dictionary<(int, int), List<ComputeBuffer>>();
    static bool _purged;
    private static ComputeBuffer _randomBuffer;

#if UNITY_EDITOR
    static BufferManager()
    {
        UnityEditor.EditorApplication.playModeStateChanged += state =>
        {
            if(state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
            {
                Purge();
            }

            if(state == UnityEditor.PlayModeStateChange.EnteredEditMode)
            {
                _purged = false;
            }
        };
    }
#endif

    public static RenderTexture AcquireTexture(int width, int height, RenderTextureFormat format, bool withMips = false)
    {
        return AcquireTexture3D(width, height, 0, format, withMips);
    }

    public static RenderTexture AcquireTexture3D(int width, int height, int depth, RenderTextureFormat format, bool withMips = false)
    {
        if(_textureLibrary.TryGetValue((width,height,depth,format,withMips), out var list))
        {
            if(list.Count != 0)
            {
                var texture = list.Last();
                list.RemoveAt(list.Count - 1);
                return texture;
            }
        }

        var output = new RenderTexture(width, height, depth, format)
        {
            enableRandomWrite = true,
            useMipMap = withMips,
            autoGenerateMips = false,
            name = "Pool Texture"
        };
        output.Create();

        if(withMips)
        {
            output.GenerateMips();
        }

        return output;
    }

    public static ComputeBuffer AcquireBuffer<T>(T[] data)
    {
        var count = data.Length;
        var structureSize = Marshal.SizeOf<T>();
        if(structureSize % 16 != 0) {
            throw new Exception($"{typeof(T).Name}'s size is {structureSize}. It must be a multiple of 16");
        }
        
        if(_bufferLibrary.TryGetValue((structureSize,count), out var list))
        {
            if(list.Count != 0)
            {
                var buffer = list.Last();
                list.RemoveAt(list.Count - 1);
                buffer.SetData(data);
                return buffer;
            }
        }

        var output = new ComputeBuffer(count, structureSize, ComputeBufferType.Structured, ComputeBufferMode.Immutable);
        output.SetData(data);
        output.name = "Pool Buffer";
        return output;
    }

    public static void Release(ref RenderTexture tex)
    {
        if(!tex)
        {
            tex = null;
            return;
        }

        if(_purged)
        {
            GameObject.DestroyImmediate(tex);
            tex = null;
            return;
        }

        var key = (tex.width, tex.height, tex.depth, tex.format, tex.useMipMap);

        List<RenderTexture> list;
        if(!_textureLibrary.TryGetValue(key, out list)) {
            list = new List<RenderTexture>();
            _textureLibrary.Add(key, list);
        }
        list.Add(tex);
        tex = null;
    }

    public static void Release(ref ComputeBuffer buffer)
    {
        if(buffer == null) return;

        if(_purged)
        {
            buffer.Dispose();
            buffer = null;
            return;
        }

        var key = (buffer.stride, buffer.count);

        List<ComputeBuffer> list;
        if(!_bufferLibrary.TryGetValue(key, out list)) {
            list = new List<ComputeBuffer>();
            _bufferLibrary.Add(key, list);
        }
        list.Add(buffer);
        buffer = null;
    }

    public static void Purge()
    {
        _purged = true;

        foreach(var list in _textureLibrary.Values)
        {
            foreach(var texture in list)
            {
                GameObject.DestroyImmediate(texture);
            }
        }

        foreach(var list in _bufferLibrary.Values)
        {
            foreach(var buffer in list)
            {
                buffer.Dispose();
            }
        }

        _textureLibrary.Clear();
        _bufferLibrary.Clear();

        if(_randomBuffer != null)
        {
            _randomBuffer.Release();
            _randomBuffer = null;
        }

        if(_mieScatteringLUT)
        {
            GameObject.DestroyImmediate(_mieScatteringLUT);
            _mieScatteringLUT = null;
        }

        if(_teardropScatteringLUT)
        {
            GameObject.DestroyImmediate(_teardropScatteringLUT);
            _teardropScatteringLUT = null;
        }

        if(_bdrfLUT)
        {
            GameObject.DestroyImmediate(_bdrfLUT); 
            _bdrfLUT = null;
        }
    }

    /////////////////////////
    // Specialty buffers
    /////////////////////////

    // Collection of random seeds for massive parallel random number generation (see Random.cginc)
    public static ComputeBuffer GetRandomSeedBuffer(int seeds)
    {
        if(_purged) return null;

        if(_randomBuffer != null && _randomBuffer.count < seeds)
        {
            _randomBuffer.Release();
            _randomBuffer = null;
        }

        if(_randomBuffer == null)
        {
            uint4[] seedcollection = new uint4[seeds];

            for (int i = 0; i < seeds; i++)
            {
                seedcollection[i].x = (uint)(UnityEngine.Random.value * 1000000);
                seedcollection[i].y = (uint)(UnityEngine.Random.value * 1000000);
                seedcollection[i].z = (uint)(UnityEngine.Random.value * 1000000);
                seedcollection[i].w = (uint)(UnityEngine.Random.value * 1000000);
            }

            _randomBuffer = AcquireBuffer(seedcollection);
            _randomBuffer.name = $"Random Seed Buffer ({seeds} seeds)";
        }

        return _randomBuffer;
    } 

    #region MieScatteringLUT
    public static Texture MieScatteringLUT
    {
        get
        {
            if(_purged) return null;
            if(_mieScatteringLUT == null)
            {
                _mieScatteringLUT = LUT.CreateMieScatteringLUT().AsTexture();
                _mieScatteringLUT.name = "Mie Scattering LUT";
            }
            return _mieScatteringLUT;
        }
    }
    private static Texture _mieScatteringLUT;
    #endregion

    #region TeardropScatteringLUT
    public static Texture TeardropScatteringLUT
    {
        get
        {
            if(_purged) return null;
            if(_teardropScatteringLUT == null)
            {
                float strength = 10;
                _teardropScatteringLUT = LUT.CreateTeardropScatteringLUT(strength).AsTexture();
                _teardropScatteringLUT.name = $"Teardrop Scattering LUT (Strength={strength})";
            }
            return _teardropScatteringLUT;
        }
    }
    private static Texture _teardropScatteringLUT;
    #endregion

    #region BRDFLUT
    public static Texture BRDFLUT
    {
        get
        {
            if(_purged) return null;
            if(_bdrfLUT == null)
            {
                _bdrfLUT = LUT.CreateBDRFLUT().AsTexture();
                _bdrfLUT.name = "BDRF LUT";
            }
            return _bdrfLUT;
        }
    }
    private static Texture _bdrfLUT;
    #endregion
}