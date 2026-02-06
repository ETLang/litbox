using UnityEngine;

[RequireComponent(typeof(Camera))]
public class ForceHDR_UE5 : MonoBehaviour
{
    [SerializeField, Range(-4, 4)] public float exposure = 0.0f;
    [SerializeField] public Vector3 whitePointLog = new Vector3(2.0f, 2.0f, 2.0f);
    [SerializeField] public Vector3 blackPointLog = new Vector3(-4.0f, -4.0f, -4.0f);

    Camera _cam;
    RenderTexture _hdrTarget;
    Material _toneMapper;

    void Start()
    {
        _cam = GetComponent<Camera>();

        _hdrTarget = new RenderTexture(_cam.pixelWidth, _cam.pixelHeight, 24, RenderTextureFormat.ARGBFloat);
        _hdrTarget.Create();

        _toneMapper = new Material(Shader.Find("Hidden/LitboxToneMapping_UE5"));
    }

    void OnPreRender()
    {
        _toneMapper.SetFloat("_Exposure", exposure);
        _toneMapper.SetVector("_WhitePointLog", whitePointLog);
        _toneMapper.SetVector("_BlackPointLog", blackPointLog);

        _cam.SetTargetBuffers(_hdrTarget.colorBuffer, _hdrTarget.depthBuffer);
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Graphics.Blit(_hdrTarget, destination, _toneMapper);
    }
}