using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

public static class TrainingUtil
{
    private static SimpleRNG _rand = new SimpleRNG(0);

    public static void Clear(this RenderTexture target, Color color) {
        RenderTexture rt = RenderTexture.active;
        RenderTexture.active = target;
        GL.Clear(true, true, color);
        RenderTexture.active = rt;
    }

    public static void SaveTextureEXR(this RenderTexture target, string path)
    {
        if(target.format != RenderTextureFormat.ARGBFloat) {
            var descriptor = target.descriptor;
            descriptor.colorFormat = RenderTextureFormat.ARGBFloat;
            descriptor.sRGB = false;
            descriptor.mipCount = 1;

            var floatTarget = new RenderTexture(descriptor);
            floatTarget.Create();

            var current = RenderTexture.active;
            Graphics.Blit(target, floatTarget);
            RenderTexture.active = current;
            target = floatTarget;
        }

        Texture2D image = new Texture2D(target.width, target.height, TextureFormat.RGBAFloat, false, true);

        Graphics.SetRenderTarget(target);
        image.ReadPixels(new Rect(0, 0, image.width, image.height), 0, 0);
        image.Apply();

        byte[] bytes = image.EncodeToEXR(Texture2D.EXRFlags.CompressZIP);
        System.IO.File.WriteAllBytes(path, bytes);
    }

    public static void SaveTexturePNG(this RenderTexture target, string path)
    {
        if(!target.sRGB) {
            var descriptor = target.descriptor;
            descriptor.colorFormat = RenderTextureFormat.ARGB32;
            descriptor.sRGB = true;
            descriptor.mipCount = 1;

            var srgbTarget = new RenderTexture(descriptor);
            srgbTarget.Create();

            var current = RenderTexture.active;
            Graphics.Blit(target, srgbTarget);
            RenderTexture.active = current;
            target = srgbTarget;
        }

        Texture2D image = new Texture2D(target.width, target.height, TextureFormat.RGBA32, false, true);

        Graphics.SetRenderTarget(target);
        image.ReadPixels(new Rect(0, 0, image.width, image.height), 0, 0);
        image.Apply();

        byte[] bytes = image.EncodeToPNG();
        System.IO.File.WriteAllBytes(path, bytes);

        GameObject.DestroyImmediate(image);
    }

    public static string GetTrainingFolder(string sessionName)
    {
        return Application.dataPath + "/" + sessionName;
    }

    public static double RandomRanged(double min, double max)
    {
        return min + _rand.NextDouble() * (max - min);
    }

    public static float RandomRanged(float min, float max)
    {
        return min + _rand.NextFloat() * (max - min);
    }

    public static Vector2 RandomRanged(Vector2 min, Vector2 max)
    {
        return min + Vector2.Scale(_rand.NextFloat2(), max - min);
    }

    public static Vector3 RandomRanged(Vector3 min, Vector3 max)
    {
        return min + Vector3.Scale(_rand.NextFloat3(), max - min);
    }

    public static Vector4 RandomRanged(Vector4 min, Vector4 max)
    {
        return min + Vector4.Scale(_rand.NextFloat4(), max - min);
    }

    public static T RandomFromSet<T>(T[] set)
    {
        return set[_rand.NextUint() % set.Length];
    }

    public static T RandomFromEnum<T>() where T : Enum
    {
        return RandomFromSet((T[])Enum.GetValues(typeof(T)));
    }

    public static Vector2 RandomDirection2D()
    {
        var angle = _rand.NextDouble() * Math.PI * 2;
        return new Vector2((float)Math.Cos(angle), (float)Math.Sin(angle));
    }

    public static float RandomStat(float mean, float stdev)
    {
        return mean + stdev * _rand.NextNormal1F();
    }

    public static Vector2 RandomStat(Vector2 mean, Vector2 stdev)
    {
        return mean + Vector2.Scale(stdev, _rand.NextNormal2F());
    }

    public static Vector3 RandomStat(Vector3 mean, Vector3 stdev)
    {
        return mean + Vector3.Scale(stdev, _rand.NextNormal3F());
    }

    public static Vector4 RandomStat(Vector4 mean, Vector4 stdev)
    {
        return mean + Vector4.Scale(stdev, _rand.NextNormal4F());
    }

    public static float RandomRangeLog(float min, float max)
    {
        if (min * max <= 0)
            throw new ArgumentOutOfRangeException("RandomRangeLog requires min and max to be the same sign and nonzero");

        var sign = Math.Sign(min);

        if (sign < 0)
        {
            var tmp = min;
            min = Math.Abs(max);
            max = Math.Abs(min);
        }

        var lnMin = Math.Log(min, Math.E);
        var lnMax = Math.Log(max, Math.E);

        return sign * (float)Math.Exp(RandomRanged(lnMax, lnMin));
    }
}
