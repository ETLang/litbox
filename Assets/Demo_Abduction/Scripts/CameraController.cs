using UnityEngine;

[RequireComponent(typeof(Camera))]
public class CameraController : MonoBehaviour
{
    [SerializeField] Collider2D thingToFollow;
    [SerializeField] float wiggleRoom = 0.5f;
    [SerializeField] float verticalSweetSpot = 0.6f;
    [SerializeField] float verticalWiggleRoom = 0.1f;
    [SerializeField] float groundY = -4.0f;
    [SerializeField] float damping = 0.8f;
    [SerializeField] float maxVelocity = 5;
    [SerializeField] float maxAcceleration = 50;

    float velocityX = 0;
    float velocityY = 0;

    void Update()
    {
        var camera = GetComponent<Camera>();

        if(thingToFollow) {
            var followX = thingToFollow.transform.position.x;
            var followY = thingToFollow.transform.position.y;

            var leftEdge = transform.position.x - camera.orthographicSize * camera.aspect;
            var rightEdge = transform.position.x + camera.orthographicSize * camera.aspect;
            var bottomEdge = transform.position.y - camera.orthographicSize;
            var topEdge = transform.position.y + camera.orthographicSize;
            var leftRoamEdge = transform.position.x - wiggleRoom * camera.orthographicSize * camera.aspect;
            var rightRoamEdge = transform.position.x + wiggleRoom * camera.orthographicSize * camera.aspect;

            var verticalFocalPoint = verticalSweetSpot * (topEdge - bottomEdge) + bottomEdge;
            var bottomRoamEdge = verticalFocalPoint - verticalWiggleRoom * camera.orthographicSize;
            var topRoamEdge = verticalFocalPoint + verticalWiggleRoom * camera.orthographicSize;

            var prevPosition = transform.position;

            float idealX = float.NaN;
            float idealY = float.NaN;
            float requiredDX = float.NaN;
            float requiredDY = float.NaN;

            var followBounds = thingToFollow.bounds;

            if(followX < leftRoamEdge) {
                if(followBounds.min.x < leftEdge) {
                    requiredDX = -(leftEdge - followBounds.min.x);
                }
                idealX = transform.position.x - (leftRoamEdge - followX);
            } else if(followX > rightRoamEdge) {
                if(followBounds.max.x > rightEdge) {
                    requiredDX = (followBounds.max.x - rightEdge);
                }
                idealX = transform.position.x + (followX - rightRoamEdge);
            }

            if(followY < bottomRoamEdge) {
                if (followBounds.min.y < bottomEdge) {
                    requiredDY = -(bottomEdge - followBounds.min.y);
                }
                idealY = transform.position.y - (bottomRoamEdge - followY);
            } else if(followY > topRoamEdge) {
                if (followBounds.max.y > topEdge) {
                    requiredDY = (followBounds.max.y - topEdge);
                }
                idealY = transform.position.y + (followY - topRoamEdge);
            }

            var groundBasedY = groundY + camera.orthographicSize;

            if(transform.position.y < groundBasedY) {
                idealY = groundBasedY;
            }

            float idealXVelocity = float.NaN;
            float idealYVelocity = float.NaN;

            if(!float.IsNaN(idealX)) {
                idealXVelocity = (idealX - transform.position.x) / Time.deltaTime;
            }

            if(!float.IsNaN(idealY)) {
                idealYVelocity = (idealY - transform.position.y) / Time.deltaTime;
            }

            var frameDamping = Mathf.Pow(1 - damping, Time.deltaTime);

            var nextVelocityX = velocityX * frameDamping;
            var nextVelocityY = velocityY * frameDamping;

            if (!float.IsNaN(idealXVelocity)) {
                if(idealXVelocity * nextVelocityX < 0) {
                    velocityX = 0;
                    nextVelocityX = 0;
                }
                var sign = Mathf.Sign(idealXVelocity);
                idealXVelocity = Mathf.Abs(idealXVelocity);
                nextVelocityX = Mathf.Abs(nextVelocityX);
                nextVelocityX = sign * Mathf.Min(maxVelocity, Mathf.Max(nextVelocityX, Mathf.Min(sign * velocityX + maxAcceleration * Time.deltaTime, idealXVelocity - sign*velocityX)));
            }

            if(!float.IsNaN(idealYVelocity)) {
                if (idealYVelocity * nextVelocityY < 0) {
                    velocityY = 0;
                    nextVelocityY = 0;
                }
                var sign = Mathf.Sign(idealYVelocity);
                idealYVelocity = Mathf.Abs(idealYVelocity);
                nextVelocityY = Mathf.Abs(nextVelocityY);
                nextVelocityY = sign * Mathf.Min(maxVelocity, Mathf.Max(nextVelocityY, Mathf.Min(sign * velocityY + maxAcceleration * Time.deltaTime, idealYVelocity - sign*velocityY)));
            }

            velocityX = nextVelocityX;
            velocityY = nextVelocityY;

            var nextX = transform.position.x + velocityX * Time.deltaTime;
            var nextY = transform.position.y + velocityY * Time.deltaTime;

            if(!float.IsNaN(requiredDX)) {
                nextX = transform.position.x + requiredDX;
            }

            if(!float.IsNaN(requiredDY)) {
                nextY = transform.position.y + requiredDY;
            }

            transform.position = new Vector3(nextX, nextY, transform.position.z);
        }
    }
}
