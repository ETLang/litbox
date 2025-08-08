using UnityEngine;
using UnityEngine.InputSystem;

public class UfoController : MonoBehaviour
{
    [SerializeField] float thrustPower = 3;
    [SerializeField] float turnPower = 4;
    //[SerializeField] float maxTurnAngle = 15;
    [SerializeField] float returnForce = 1;

    bool _thrusterEngaged = false;
    float _rotationControl = 0;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        
    }

    void FixedUpdate()
    {
        var body = GetComponent<Rigidbody2D>();

        if (_thrusterEngaged) {
            //body.WakeUp();
            body.AddRelativeForceY(thrustPower);
        }

        var angle = body.rotation;

        if (angle > 180) {
            angle = 180 - angle;
        }

        var returnTorque = Mathf.Pow(Mathf.Abs(angle), 2) * returnForce * -Mathf.Sign(angle);
        var desiredTorque = turnPower * -_rotationControl;

        if (desiredTorque != 0) {
            //body.WakeUp();
        }

        body.AddTorque(desiredTorque + returnTorque);
    }

    void OnMove(InputValue valueIn)
    {
        var valueObj = valueIn.Get();
        float value = 0.0f;
        if (valueObj != null)
            value = (float)valueObj;

        _rotationControl = value;
    }

    void OnThrust(InputValue valueIn)
    {
        _thrusterEngaged = valueIn.isPressed;
    }

    void OnTractor(InputValue valueIn)
    {
        Debug.Log("INPUT: Tractor - " + valueIn.isPressed.ToString());
    }

    void OnDeviceLost()
    {
        Debug.Log("INPUT: Device Lost");
    }

    void OnDeviceRegained()
    {
        Debug.Log("INPUT: Device Regained");
    }

    void OnControlsChanged()
    {
        Debug.Log("INPUT: Controls Changed");
    }
}
