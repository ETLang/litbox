using UnityEngine;
using UnityEngine.InputSystem.LowLevel;

public class GenerateGradientTexture : GeneratorBase<Texture2D>
{
    [SerializeField] Gradient gradient = new Gradient();
    [SerializeField] TextureWrapMode wrapMode = TextureWrapMode.Clamp;
    [SerializeField] bool hdr;
    [SerializeField, ReadOnly] Texture2D texture;

    GenerateGradientTexture()
    {
        DetectChanges(() => gradient);
        DetectChanges(() => wrapMode);
    }

    public override Object CreateAsset()
    {
        return new Texture2D(1, 2048, hdr ? TextureFormat.RGBAFloat : TextureFormat.RGBA32, true);
    }

    public override void Populate(Object asset)
    {
        texture = (Texture2D)asset;
        texture.Reinitialize(1, 2048, hdr ? TextureFormat.RGBAFloat : TextureFormat.RGBA32, true);

        texture.wrapMode = wrapMode;
        for (int i = 0; i < texture.height; i++) {
            Color color = gradient.Evaluate((float)i / (texture.height - 1));
            texture.SetPixel(0, i, color);
        }
        texture.Apply();
    }
}
