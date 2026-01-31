using System.Collections.Specialized;
using UnityEngine;

public class GenerateCloudTexture : GeneratorBase<Texture2D>
{
    [SerializeField] int textureSize = 512;
    [SerializeField] float curvePower = 8.0f;
    [SerializeField, Range(0,1)] float easing = 0;
    [SerializeField, ReadOnly] Texture2D texture;

    GenerateCloudTexture()
    {
        DetectChanges(() => textureSize);
    }

    public override Object CreateAsset()
    {
        return new Texture2D(textureSize, textureSize, TextureFormat.RGBA32, true);
    }

    private float ess(float x) => x * x * (3 - 2 * x);

    public override void Populate(Object asset)
    {
        texture = (Texture2D)asset;
        texture.Reinitialize(textureSize, textureSize, TextureFormat.RGBA32, true);

        for (int y = 0; y < texture.height; y++) {
            for (int x = 0; x < texture.width; x++) {
                float xCoord = (float)x / texture.width * 2 - 1;
                float yCoord = (float)y / texture.height * 2 - 1;

                Vector2 v = new Vector2(xCoord, yCoord);

                //float f = (v.magnitude < 1) ? 1 : 0;
                float f = 1-Mathf.Pow(Mathf.Clamp(v.magnitude, 0, 1), curvePower);
                //float f = 1-ess(Mathf.Clamp(v.magnitude, 0, 1));
                    
                Color color = new Color(1, 1, 1, Mathf.Lerp(f, ess(f), easing));
                texture.SetPixel(x, y, color);
            }
        }
        texture.Apply();
    }
}