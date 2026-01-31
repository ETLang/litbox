using UnityEngine;

[RequireComponent(typeof(Simulation))]
public class BindSimulationToCamera : MonoBehaviour
{
    [SerializeField, Range(0.0625f, 1)] float resolutionScale = 0.25f;
    [SerializeField, Range(0, 100)] float paddingPercent = 0;

    private Simulation _sim;
    private Camera _cam;

    public Matrix4x4 ScreenToSimulationUVTransform { get; private set; }

    private void Awake()
    {
        _sim = GetComponent<Simulation>();
        _cam = transform.parent.GetComponent<Camera>();

        if(_cam == null) {
            Debug.LogError("BindSimulationCamera requires a camera to be its direct parent");
        } else if(!_cam.orthographic) {
            Debug.LogWarning("Camera is not orthographic");
        } else if(_cam.transform.position.z != 0) {
            Debug.LogWarning("Camera Z is not zero");
        }
    }

    // Update is called once per frame
    void Update()
    {
        float padding = paddingPercent / 100.0f;

        _sim.width = (int)((_cam.pixelWidth + 2 * _cam.pixelHeight * padding) * resolutionScale);
        _sim.height = (int)((_cam.pixelHeight + 2 * _cam.pixelHeight * padding) * resolutionScale);

        var cameraScale = _cam.transform.lossyScale;
        var xPaddingScale = 1.0f + 2 * padding * _cam.pixelHeight / _cam.pixelWidth;
        var yPaddingScale = 1.0f + 2 * padding;
        _sim.transform.localScale = new Vector3(
            xPaddingScale * _cam.orthographicSize * _cam.aspect * 2 / cameraScale.x, 
            yPaddingScale * _cam.orthographicSize * 2 / cameraScale.y, 1);

        ScreenToSimulationUVTransform =
            Matrix4x4.Translate(new Vector3(0.5f, -0.5f, 0)) *
            Matrix4x4.Scale(new Vector3(0.5f / xPaddingScale, -0.5f / yPaddingScale));
    }
}
