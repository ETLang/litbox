using System;
using System.Reflection;
using UnityEngine;

public class TracerPostProcessor : Disposable
{
    private ComputeShader _postProcessingShader;
    private int[] _computeCVAndMipsKernel;
    private int[] _generateMipsKernel;

    private static TracerPostProcessor _Instance;
    public static TracerPostProcessor Instance =>
        _Instance ?? (_Instance = new TracerPostProcessor());

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

    private TracerPostProcessor()
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

    protected override void OnDispose()
    {
        _Instance = null;
        base.OnDispose();
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

        GenerateMips(destMean, firstDispatchMipCount - 1);
    }

    public void GenerateMips(RenderTexture texture, int detailLevel = 0, int mipCount = 0)
    {
        if(mipCount == 0)
        {
            mipCount = texture.mipmapCount - 1 - detailLevel;
        }

        int lastMip = detailLevel + mipCount;
        while(detailLevel < lastMip)
        {
            int nextDispatchMipCount = Mathf.Min(lastMip - detailLevel, 4);
            int nextDispatchKernel = _generateMipsKernel[nextDispatchMipCount - 1];

            _postProcessingShader.SetTexture(nextDispatchKernel, _SourceAId, texture, detailLevel);

            for(int i = 1;i <= nextDispatchMipCount;i++) {
                _postProcessingShader.SetTexture(nextDispatchKernel, _OutMipId[i], texture, detailLevel + i);
            }
            _postProcessingShader.Dispatch(nextDispatchKernel, (texture.MipWidth(detailLevel) - 1) / 16 + 1, (texture.MipHeight(detailLevel) - 1) / 16 + 1, 1);

            detailLevel += nextDispatchMipCount;
        }
    }
}