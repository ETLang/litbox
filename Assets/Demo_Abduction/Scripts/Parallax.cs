using UnityEngine;

public class Parallax : MonoBehaviour
{
    [SerializeField] float parallaxRateX;
    [SerializeField] float parallaxRateY;

    Vector3 initialDelta;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        initialDelta = transform.position - Camera.main.transform.position;
    }

    // Update is called once per frame
    void LateUpdate()
    {
        var cameraPos = Camera.main.transform.position;

        cameraPos.x *= (1 - parallaxRateX);
        cameraPos.y *= (1 - parallaxRateY);

        transform.position = cameraPos + initialDelta;
    }
}
