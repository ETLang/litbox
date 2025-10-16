using UnityEngine;

[RequireComponent(typeof(Simulation))]
public class SimulationSizeFromCamera : MonoBehaviour
{
    [SerializeField] float ratio = 0.25f;
    [SerializeField] Transform scaleObjectWithAspect = null;

    Simulation _sim;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        _sim = GetComponent<Simulation>();
    }

    // Update is called once per frame
    void Update()
    {
        _sim.width = (int)(Camera.main.pixelWidth * ratio);
        _sim.height = (int)(Camera.main.pixelHeight * ratio);

        if (scaleObjectWithAspect != null) {
            scaleObjectWithAspect.localScale = new Vector3(Camera.main.aspect, 1, 1);
        }
    }
}
