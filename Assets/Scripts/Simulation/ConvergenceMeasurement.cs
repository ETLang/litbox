using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using Unity.Collections;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

public class ConvergenceMeasurement : Disposable
{
    ComputeShader _convergenceShader;
    ComputeBuffer _totalVariance;
    uint4[] _clearValue = new uint4[] { new uint4(0) };
    int _computeConvergenceKernel;

    static int _VarianceId = Shader.PropertyToID("_variance");
    static int _TotalVarianceId = Shader.PropertyToID("_totalVariance");

    public ConvergenceMeasurement()
    {
        _convergenceShader = (ComputeShader)Resources.Load("Convergence");
        _totalVariance = this.CreateStructuredBuffer(_clearValue);
        _totalVariance.name = "Total Computed Variance";
        _computeConvergenceKernel = _convergenceShader.FindKernel("ComputeConvergence");

        _convergenceShader.SetBuffer(_computeConvergenceKernel, _TotalVarianceId, _totalVariance);
    }

    Task<float> _existingRequest;
    public Task<float> GetVarianceAsync(Texture varianceMap)
    {
        if(_existingRequest == null)
        {
            _existingRequest = GetVarianceInternal(varianceMap);
        }

        return _existingRequest;
    }

    private async Task<float> GetVarianceInternal(Texture varianceMap)
    {
        _totalVariance.SetData(_clearValue);
        _convergenceShader.SetTexture(_computeConvergenceKernel, _VarianceId, varianceMap);
        _convergenceShader.DispatchAutoGroup(_computeConvergenceKernel, varianceMap.width, varianceMap.height, 1);
        var feedback = await AsyncGPUReadback.RequestAsync(_totalVariance);
        var dataBlock = feedback.GetData<uint>();
        var rawValue = dataBlock[0];

        _existingRequest = null;

        if(!varianceMap)
        {
            return -1;
        }
        
        return rawValue / (10000.0f * varianceMap.width * varianceMap.height);
    }
}