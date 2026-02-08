using UnityEngine;

public class AnalysisParameters : MonoBehaviour
{
    [SerializeField, Range(0.01f, 5f)]
    public float SigmaSpatial = 1.2f;
    [SerializeField, Range(0.01f, 0.2f)]
    public float SigmaAlbedo = 0.05f;
    [SerializeField, Range(0.01f, 1f)]
    public float SigmaLuminanceTight = 0.05f;
    [SerializeField, Range(1f, 5f)]
    public float SigmaLuminanceLoose = 2.5f;
    [SerializeField, Range(0.1f, 10f)]
    public float KLuminance = 2.0f;
}