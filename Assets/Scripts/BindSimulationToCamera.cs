using UnityEngine;

[RequireComponent(typeof(Simulation))]
public class BindSimulationToCamera : MonoBehaviour
{
    [SerializeField] float resolutionScale = 0.25f;

    private Simulation _sim;
    private Camera _cam;

    public static BindSimulationToCamera Main { get; private set; }

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

        if(_cam == Camera.main) {
            Main = this;
        }
    }

    void OnDestroy()
    {
        if(Main == this) {
            Main = null;
        }
    }

    // Update is called once per frame
    void Update()
    {
        _sim.width = (int)(_cam.pixelWidth * resolutionScale);
        _sim.height = (int)(_cam.pixelHeight * resolutionScale);

        var cameraScale = _cam.transform.lossyScale;
        _sim.transform.localScale = new Vector3(_cam.orthographicSize * _cam.aspect * 2 / cameraScale.x, _cam.orthographicSize * 2 / cameraScale.y, 1);
    }
}
