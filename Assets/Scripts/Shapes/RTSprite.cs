using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(SpriteRenderer))]
public class RTSprite : RTObject
{
    private Sprite _previousSprite;
    private Color _previousColor;

    new protected void Start() {
        base.Start();
        
        var spriteRenderer = GetComponent<SpriteRenderer>();
        spriteRenderer.material.color = spriteRenderer.color;

        _previousSprite = spriteRenderer.sprite;
        _previousColor = spriteRenderer.color;
    }
    
    new protected void Update()
    {
        base.Update();

        var spriteRenderer = GetComponent<SpriteRenderer>();

        Changed = Changed ||
            _previousSprite != spriteRenderer.sprite ||
            _previousColor != spriteRenderer.color;

        _previousSprite = spriteRenderer.sprite;
        _previousColor = spriteRenderer.color;

        if(Changed) {
            spriteRenderer.material.color = spriteRenderer.color;
        }
    }
}
