using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using UnityEngine;
using UnityEngine.XR;

public class StarController : MonoBehaviour
{
    [SerializeField] GameObject brightStarPrefab;
    [SerializeField] GameObject normalStarPrefab;

    [SerializeField] int starDensity = 200;
    [SerializeField] float percentBrightStars = 10;

    [SerializeField] float normalStarMinScale = 0.5f;
    [SerializeField] float normalStarMaxScale = 1.1f;
    [SerializeField] float normalStarMinBrightness = 0.7f;

    [SerializeField] float brightStarMinScale = 0.9f;
    [SerializeField] float brightStarMaxScale = 1.2f;

    const float _BlockSize = 10;

    List<GameObject> _instances = new List<GameObject>();
    Dictionary<(int, int), int> _blockOffsets = new Dictionary<(int, int), int>();
    Stack<int> _unusedOffsets = new Stack<int>();

    private void Start()
    {
    }

    private void Update()
    {
        var camera = Camera.main;
        var cameraPos = camera.transform.position - transform.position;
        var cameraMinX = cameraPos.x - 2 * camera.orthographicSize * camera.aspect;
        var cameraMinY = cameraPos.y - 2 * camera.orthographicSize;
        var cameraMaxX = cameraPos.x + 2 * camera.orthographicSize * camera.aspect;
        var cameraMaxY = cameraPos.y + 2 * camera.orthographicSize;

        var blockMinX = (int)Mathf.Floor(cameraMinX / _BlockSize);
        var blockMinY = (int)Mathf.Floor(cameraMinY / _BlockSize);
        var blockMaxX = (int)Mathf.Floor(cameraMaxX / _BlockSize);
        var blockMaxY = (int)Mathf.Floor(cameraMaxY / _BlockSize);

        var toFree = new List<(int, int)>();

        foreach(var key in _blockOffsets.Keys) {
            if (key.Item1 < blockMinX || key.Item1 > blockMaxX ||
                key.Item2 < blockMinY || key.Item2 > blockMaxY) {
                toFree.Add(key);
            }
        }

        foreach(var key in toFree) {
            _unusedOffsets.Push(_blockOffsets[key]);
            _blockOffsets.Remove(key);
        }

        for(int x = blockMinX;x <= blockMaxX;x++) {
            for(int y = blockMinY;y <= blockMaxY;y++) {
                if (!_blockOffsets.ContainsKey((x, y))) {
                    AllocateBlock(x, y);
                }
            }
        }
    }

    private void AllocateBlock(int x, int y)
    {
        var proportionBrightStars = percentBrightStars / 100.0f;

        if(_unusedOffsets.Count == 0) {
            _blockOffsets[(x, y)] = _instances.Count;

            int lastBright = (int)((float)starDensity * proportionBrightStars);

            for(int i = 0;i < starDensity;i++) {
                GameObject newStar;
                float scale;
                float brightness;

                if(i < lastBright) {
                    newStar = Instantiate(brightStarPrefab, transform);
                    scale = Random.Range(brightStarMinScale, brightStarMaxScale);
                    brightness = 1;
                } else {
                    newStar = Instantiate(normalStarPrefab, transform);
                    scale = Random.Range(normalStarMinScale, normalStarMaxScale);
                    brightness = Random.Range(normalStarMinBrightness, 1);
                }

                newStar.GetComponentInChildren<SpriteRenderer>().color = new Color(1, 1, 1, brightness);
                newStar.transform.localScale = new Vector3(scale, scale, 1);

                _instances.Add(newStar);
            }
        } else {
            _blockOffsets[(x, y)] = _unusedOffsets.Pop();
        }

        SetupBlock(x, y);
    }

    private void SetupBlock(int x, int y)
    {
        var rand = new System.Random(x + y * 107);
        var offset = _blockOffsets[(x, y)];

        for(int i = 0;i < starDensity;i++) {
            var star = _instances[i + offset];
            float starX = (rand.NextSingle() + (float)x) * _BlockSize;
            float starY = (rand.NextSingle() + (float)y) * _BlockSize;

            star.transform.localPosition = new Vector3(starX, starY, transform.localPosition.z);
        }
    }
}
