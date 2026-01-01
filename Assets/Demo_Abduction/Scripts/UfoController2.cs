using System.Linq;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Processors;

/* 

V2 controls:
- UFO hovers without any input from user; gravity does not cause it to fall.
- WASD moves
- Possible ground impact avoidance
WASD is an expression of intent. The UFO will try to move in that direction, but will not do so instantly.
The UFO will automatically stabilize its rotation to be upright when there is no rotational input.
The UFO will accelerate in the direction of movement input, up to a maximum speed.

 */
public class UfoController2 : MonoBehaviour
{
    [SerializeField] float maxSpeed = 5;
    [SerializeField] float horizontalAcceleration = 10;
    [SerializeField] float verticalAcceleration = 10;
    [SerializeField] float maxAltitude = 20;
    [SerializeField] float maxTiltAngle = 15;
    [SerializeField] float returnForce = 1;


    float _xIntent = 0;
    float _yIntent = 0;

    PlayerInput _input;

    private void Awake()
    {
        // what does this do again?
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
    
    void Start()
    {
    }

    void FixedUpdate()
    {
        var body = GetComponent<Rigidbody2D>();

        // if(body.IsSleeping()) {
        //     return;
        // }

        float intendedXVelocity = _xIntent * maxSpeed;
        float intendedYVelocity = _yIntent * maxSpeed;
        float deltaX = intendedXVelocity - body.linearVelocity.x;
        float deltaY = intendedYVelocity - body.linearVelocity.y;

        float dxSign = Mathf.Sign(deltaX);
        deltaX = Mathf.Abs(deltaX);
        float dySign = Mathf.Sign(deltaY);
        deltaY = Mathf.Abs(deltaY);

        float accelX = Mathf.Clamp(horizontalAcceleration /* Time.fixedDeltaTime*/, 0, deltaX) * dxSign;
        float accelY = Mathf.Clamp(verticalAcceleration /* Time.fixedDeltaTime*/, 0, deltaY) * dySign;

        body.AddForce(new Vector2(accelX, accelY) * body.mass, ForceMode2D.Force);

        var angle = body.rotation;

        if (angle > 180) {
            angle = 180 - angle;
        }

        var returnTorque = Mathf.Pow(Mathf.Abs(angle), 2) * returnForce * -Mathf.Sign(angle);
        var desiredTorque = maxTiltAngle * -body.linearVelocity.x;

        body.AddTorque(desiredTorque + returnTorque);
    }

    void OnMove(InputValue valueIn)
    {
        var valueObj = valueIn.Get();
        float value = 0.0f;
        if (valueObj != null)
            value = (float)valueObj;
        _xIntent = value;
    }

    void OnThrust2(InputValue valueIn)
    {
        var valueObj = valueIn.Get();
        float value = 0.0f;
        if (valueObj != null)
            value = (float)valueObj;
        _yIntent = value;

        // var body = GetComponent<Rigidbody2D>();
        // if (body.IsTouchingLayers()) {
        //     body.MovePosition(transform.position + new Vector3(0, 0.00001f, 0));
        // }
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
