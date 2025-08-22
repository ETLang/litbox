using System.Linq;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Processors;

public class UfoController : MonoBehaviour
{
    [SerializeField] float thrustPower = 3;
    [SerializeField] float turnPowerDesktop = 4;
    [SerializeField] float turnPowerMobile = 4;
    //[SerializeField] float maxTurnAngle = 15;
    [SerializeField] float returnForce = 1;

    bool _thrusterEngaged = false;
    float _rotationControl = 0;
    float _turnPower;

    PlayerInput _input;

    private void Awake()
    {
        var playerinput = GetComponent<PlayerInput>();
        var move_action = playerinput.actions.Single(a => a.name.Contains("Move"));
        move_action.Disable();
        var touchscreen_move = move_action.bindings.Single(b => b.path.Contains("Touchscreen"));
        var processor_str = $"Normalize(min=0,max={Screen.width},zero={Screen.width / 2})";
        touchscreen_move.overrideProcessors = processor_str;
        move_action.ApplyBindingOverride(move_action.bindings.Count - 1, touchscreen_move);
        move_action.Enable();
    }

    void OnCollisionEnter2D(Collision2D collision)
    {
        // anything to do? Probably soon.
    }
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        _turnPower = MobileDetector.IsMobileDevice ? turnPowerMobile : turnPowerDesktop;
    }

    void FixedUpdate()
    {
        var body = GetComponent<Rigidbody2D>();

        if(body.IsSleeping()) {
            return;
        } 

        if (_thrusterEngaged) {
            //body.WakeUp();
            body.AddRelativeForceY(thrustPower);
        }

        var angle = body.rotation;

        if (angle > 180) {
            angle = 180 - angle;
        }

        var returnTorque = Mathf.Pow(Mathf.Abs(angle), 2) * returnForce * -Mathf.Sign(angle);
        var desiredTorque = _turnPower * -_rotationControl;

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

        var body = GetComponent<Rigidbody2D>();
        if (body.IsTouchingLayers()) {
            body.MovePosition(transform.position + new Vector3(0, 0.00001f, 0));
        }
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

    private void OnPause()
    {
        if (GameStateController.Instance.State != GameStates.Paused) {
            GameStateController.Instance.State = GameStates.Paused;
        } else {
            GameStateController.Instance.State = GameStates.Playing;
        }
    }
}
