using UnityEngine;

public class Parallax : MonoBehaviour
{
    [SerializeField] float parallaxRateX;
    [SerializeField] float parallaxRateY;

    Vector3 previousCameraPosition;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        //initialDelta = transform.position - Camera.main.transform.position;
    }

    // Update is called once per frame
    void LateUpdate()
    {
        var cameraPos = Camera.main.transform.position;

        var cameraDelta = cameraPos - previousCameraPosition;

        cameraDelta.x *= (1 - parallaxRateX);
        cameraDelta.y *= (1 - parallaxRateY);

        transform.position += cameraDelta;
        previousCameraPosition = cameraPos;
    }
}
