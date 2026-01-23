using System;
using System.Reflection;
using UnityEngine;

public class TracerPostProcessor : Disposable
{
    private ComputeShader _postProcessingShader;
    private int[] _computeCVAndMipsKernel;
    private int[] _generateMipsKernel;

    private static int _SourceAId = Shader.PropertyToID("_sourceA");
    private static int _SourceBId = Shader.PropertyToID("_sourceB");
    private static int _OutCVId = Shader.PropertyToID("_out_cv");
    private static int[] _OutMipId = new int[]
    {
        Shader.PropertyToID("_out_mip0"),
        Shader.PropertyToID("_out_mip1"),
        Shader.PropertyToID("_out_mip2"),
        Shader.PropertyToID("_out_mip3"),
        Shader.PropertyToID("_out_mip4")
    };

    public TracerPostProcessor()
    {
        _postProcessingShader = (ComputeShader)Resources.Load("TracerPostProcessing");

        _computeCVAndMipsKernel = new int[]
        {
            _postProcessingShader.FindKernel("ComputeCVAndOneMipFromSamplePair"),
            _postProcessingShader.FindKernel("ComputeCVAndTwoMipsFromSamplePair"),
            _postProcessingShader.FindKernel("ComputeCVAndThreeMipsFromSamplePair"),
            _postProcessingShader.FindKernel("ComputeCVAndFourMipsFromSamplePair"),
            _postProcessingShader.FindKernel("ComputeCVAndFiveMipsFromSamplePair"),
        };

        _generateMipsKernel = new int[]
        {
            _postProcessingShader.FindKernel("GenerateOneMip"),
            _postProcessingShader.FindKernel("GenerateTwoMips"),
            _postProcessingShader.FindKernel("GenerateThreeMips"),
            _postProcessingShader.FindKernel("GenerateFourMips"),
        };
    }

    public void ComputeCVAndMips(RenderTexture sourceA, RenderTexture sourceB, RenderTexture destMean, RenderTexture destCV)
    {
        int totalMips = destMean.mipmapCount;

        int firstDispatchMipCount = Mathf.Min(totalMips, 5);
        int firstDispatchKernel = _computeCVAndMipsKernel[firstDispatchMipCount - 1];
        
        _postProcessingShader.SetTexture(firstDispatchKernel, _SourceAId, sourceA);
        _postProcessingShader.SetTexture(firstDispatchKernel, _SourceBId, sourceB);
        _postProcessingShader.SetTexture(firstDispatchKernel, _OutCVId, destCV);

        for(int i = 0;i < firstDispatchMipCount;i++) {
            _postProcessingShader.SetTexture(firstDispatchKernel, _OutMipId[i], destMean, i);
        }
        _postProcessingShader.Dispatch(firstDispatchKernel, (destMean.width - 1) / 16 + 1, (destMean.height - 1) / 16 + 1, 1);

        int generatedMips = firstDispatchMipCount;
        while(generatedMips < totalMips)
        {
            int nextDispatchMipCount = Mathf.Min(totalMips - generatedMips, 4);
            int nextDispatchKernel = _generateMipsKernel[nextDispatchMipCount - 1];

            _postProcessingShader.SetTexture(nextDispatchKernel, _SourceAId, destMean, generatedMips - 1);

            for(int i = 0;i < nextDispatchMipCount;i++) {
                _postProcessingShader.SetTexture(nextDispatchKernel, _OutMipId[i + 1], destMean, generatedMips + i);
            }
            _postProcessingShader.Dispatch(nextDispatchKernel, (destMean.MipWidth(generatedMips - 1) - 1) / 16 + 1, (destMean.MipHeight(generatedMips - 1) - 1) / 16 + 1, 1);

            generatedMips += nextDispatchMipCount;
        }
    }
}