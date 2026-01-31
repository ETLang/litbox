using UnityEngine;

public class PassiveRotator : MonoBehaviour
{
    [SerializeField] Vector3 axis;
    [SerializeField] float rate;

    void Update()
    {
        transform.localRotation *= Quaternion.AngleAxis(rate * Time.deltaTime, axis);
    }
}
