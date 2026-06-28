using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using UnityEngine;

[Serializable]
public struct ComputeShaderMapping
{
    public string Name;
    public ComputeShader Shader;
}

public struct DispatchSize
{
    public int x;
    public int y;
    public int z;

    public static implicit operator DispatchSize(int x)
    {
        return new DispatchSize { x = x, y = 1, z = 1 };
    }

    public static implicit operator DispatchSize(ValueTuple<int,int> t)
    {
        return new DispatchSize { x = t.Item1, y = t.Item2, z = 1 };
    }

    public static implicit operator DispatchSize(ValueTuple<int,int,int> t)
    {
        return new DispatchSize { x = t.Item1, y = t.Item2, z = t.Item3 };
    }
}

public struct TextureView
{
    public Texture texture;
    public int mipLevel;

    public static implicit operator TextureView(Texture t)
    {
        return new TextureView { texture = t, mipLevel = -1};
    }

    public static implicit operator TextureView(ComputeShaderExtensions.MipSpec spec)
    {
        return new TextureView { texture = spec.texture, mipLevel = spec.mipLevel };
    }
}

[CreateAssetMenu(fileName = "GPU Ops", menuName = "Litbox/Shader Registry")]
public class GpuOps : ScriptableObject
{
    // The list is necessary for Unity's serializer
    [SerializeField] 
    private List<ComputeShaderMapping> _serializableList = new List<ComputeShaderMapping>();
    private Dictionary<string, ComputeShader> _shaders;

    private static GpuOps _Instance;
    private static GpuOps Instance => _Instance ?? (_Instance = Resources.Load<GpuOps>("GpuOps"));

    public static ComputeShader Get(string name) => Instance.InternalGet(name);

    private void Initialize()
    {
        if (_shaders != null) return;
        
        _shaders = new Dictionary<string, ComputeShader>();
        foreach (var entry in _serializableList)
        {
            if (entry.Shader != null && !_shaders.ContainsKey(entry.Name))
            {
                _shaders.Add(entry.Name, entry.Shader);
            }
        }
    }

    private ComputeShader InternalGet(string name)
    {
        if (_shaders == null) Initialize();
        
        if (_shaders.TryGetValue(name, out ComputeShader shader))
        {
            return shader;
        }
        
        Debug.LogError($"[Litbox] Compute Shader '{name}' not found in GpuOps.");
        return null;
    }

    private static (string, object)[] PackThem(string[] names, params object[] values)
    {
        Debug.Assert(names.Length == values.Length);
        var pairs = new (string, object)[names.Length];
        for (int i = 0; i < names.Length; i++)
        {
            pairs[i] = (names[i], values[i]);
        }
        return pairs;
    }

    public static DelegateType GetCallable<DelegateType>(string id, string kernel = "CSMain") where DelegateType : Delegate
    {
        var shader = Get(id);
        if(shader == null) throw new ArgumentException("Shader not found");
        var kernelId = shader.FindKernel(kernel);

        var invokeMethod = typeof(DelegateType).GetMethod("Invoke");
        var parameters = invokeMethod.GetParameters();

        // 1. Create expressions for all input parameters
        var paramExprs = parameters.Select(p => Expression.Parameter(p.ParameterType, p.Name)).ToArray();
        var parameterNames = parameters.Skip(1).Select(p => p.Name).ToArray();

        // 2. Extract sz.x and sz.y
        var p0 = paramExprs[0]; // DispatchSize sz
        var p0x = Expression.Field(p0, "x");
        var p0y = Expression.Field(p0, "y");

        // 3. Create an object array and box the remaining parameters
        var objectArray = Expression.NewArrayInit(typeof(object), 
            paramExprs.Skip(1).Select(p => Expression.Convert(p, typeof(object))));

        // 4. Call PackThem(parameterNames, objectArray)
        var packThemMethod = typeof(GpuOps).GetMethod("PackThem", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        var packThemCall = Expression.Call(packThemMethod, Expression.Constant(parameterNames), objectArray);

        // 5. Call ComputeShaderExtensions.RunKernel(...)
        var runKernelMethod = typeof(ComputeShaderExtensions).GetMethod("RunKernel", 
            new Type[] { typeof(ComputeShader), typeof(int), typeof(int), typeof(int), typeof(ValueTuple<string, object>[]) });

        var runKernelCall = Expression.Call(runKernelMethod, 
            Expression.Constant(shader), 
            Expression.Constant(kernelId), 
            p0x, p0y, packThemCall);

        // 6. Compile into the requested delegate type
        var lambda = Expression.Lambda<DelegateType>(runKernelCall, paramExprs);
        return lambda.Compile();
    }
}