using UnityEngine;

[RequireComponent(typeof(Camera))]
public class ForceHDR_Uchimura : MonoBehaviour
{
    [SerializeField, Range(-3,3)] public float exposure = 0.0f;
    [SerializeField, Range(0, 5)] public float contrast = 1.0f;
    [SerializeField, Range(0, 1)] public float linearBase = 0.22f;
    [SerializeField, Range(0, 1)] public float linearSpan = 0.4f;
    [SerializeField] public Vector3 blackTightness = new Vector3(1.33f, 1.33f, 1.33f);
    [SerializeField] public Vector3 blackPedestal = new Vector3(0,0,0);
    [SerializeField] public float maximumBrightness = 1;

    Camera _cam;
    RenderTexture _hdrTarget;
    Material _toneMapper;
    RenderTexture _original;


    void Start()
    {
        _cam = GetComponent<Camera>();

        _hdrTarget = new RenderTexture(_cam.pixelWidth, _cam.pixelHeight, 24, RenderTextureFormat.ARGBFloat);
        _hdrTarget.Create();

        _toneMapper = new Material(Shader.Find("Hidden/PhotonerToneMapping_Uchimura"));
    }

    void OnPreRender()
    {
        _toneMapper.SetFloat("_Exposure", Mathf.Pow(10.0f, exposure));
        _toneMapper.SetFloat("_Contrast", contrast);
        _toneMapper.SetFloat("_LinearBase", linearBase);
        _toneMapper.SetFloat("_LinearSpan", linearSpan);
        _toneMapper.SetVector("_BlackTightness", blackTightness);
        _toneMapper.SetVector("_BlackPedestal", blackPedestal);
        _toneMapper.SetFloat("_MaximumBrightness", maximumBrightness);

        _original = _cam.targetTexture;
        _cam.SetTargetBuffers(_hdrTarget.colorBuffer, _hdrTarget.depthBuffer);
    }

    void OnPostRender()
    {
        _cam.targetTexture = _original;
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Graphics.Blit(_hdrTarget, destination, _toneMapper);
    }
}