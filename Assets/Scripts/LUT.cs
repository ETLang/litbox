using System;
using System.Linq;
using UnityEngine;
using Unity.Mathematics;

static class LUT
{
    public static float[] GenerateFunctionTable(Func<float,float> fn, float minima, float maxima, int samples = 2048) {
        var table = new float[samples];

        for(int i = 0;i < samples;i++) {
            float x = minima + (maxima - minima) * i / (float)(samples - 1);
            table[i] = fn(x);
        }

        return table;
    }

    public static float NormalizeDistribution(float[] distribution, float[] outNormalized) {
        float sum = distribution.Sum();

        if(sum == 0) return 0;

        for(int i = 0;i < distribution.Length;i++) {
            outNormalized[i] = distribution[i] / sum;
        }

        return sum / distribution.Length;
    }

    public static void IntegrateDistribution(float[] distribution, float[] outIntegral) {
        float accum = 0;

        for(int i = 0;i < distribution.Length;i++) {
            outIntegral[i] = distribution[i] + accum;
            accum = outIntegral[i];
        }
    }

    public static bool Invert(float[] function, float domainStart, float domainEnd, float[] outInverse, out float inverseStart, out float inverseEnd) {
        inverseStart = function.Min();
        inverseEnd = function.Max();

        for(int i = 1;i < function.Length;i++) {
            if(function[i-1] > function[i])
                return false; // Nonmonotonic function, can't invert.
        }

        for(int i = 0;i < outInverse.Length;i++) {
            outInverse[i] = inverseStart - 1;
        }

        int lastOutIndex = 0;

        for(int i = 0;i < function.Length;i++) {
            float x = domainStart + (i * (domainEnd - domainStart)) / (float)(function.Length - 1);
            float y = function[i];

            float u = (y - inverseStart) / (inverseEnd - inverseStart);

            var nearestOutIndex = (int)Math.Round(u * (outInverse.Length - 1));

            outInverse[nearestOutIndex] = x;

            for(int j = lastOutIndex + 1;j < nearestOutIndex;j++) {
                float tween = outInverse[lastOutIndex] + 
                    (x - outInverse[lastOutIndex]) * (float)(j - lastOutIndex) / (float)(nearestOutIndex - lastOutIndex);

                outInverse[j] = x;//tween;
            }

            lastOutIndex = nearestOutIndex;
        }

        return true;
    }

    public static void AngleFunctionToVectorFunction(float[] angleFunction, float2[] outVectorFunction) {
        for(int i = 0;i < angleFunction.Length;i++) {
            float angle = angleFunction[i];
            var x = Mathf.Cos(angle);
            var y = Mathf.Sin(angle);
            outVectorFunction[i] = new float2(x, y);
        }
    }

    public static void ComponentwiseGlue(float2[] xy, float[] z, float3[] outGlued) {
        for(int i = 0;i < outGlued.Length;i++) {
            outGlued[i] = new float3(xy[i], z[i]);
        }
    }

    public static Texture LoadLUTAsTexture(float[] lut) {
        var texture = new Texture2D(lut.Length, 1, TextureFormat.RFloat, false, true);
        texture.SetPixels(lut.Select(x => new Color(x, 0, 0, 0)).ToArray());
        texture.Apply();
        return texture;
    }

    public static Texture LoadLUTAsTexture(float2[] lut) {
        var texture = new Texture2D(lut.Length, 1, TextureFormat.RGFloat, false, true);
        texture.SetPixels(lut.Select(v => new Color(v.x, v.y, 0, 0)).ToArray());
        texture.Apply();
        return texture;
    }

    public static Texture LoadLUTAsTexture(float3[] lut) {
        var texture = new Texture2D(lut.Length, 1, TextureFormat.RGBAFloat, false, true);
        texture.SetPixels(lut.Select(v => new Color(v.x, v.y, v.z, 0)).ToArray());
        texture.Apply();
        return texture;
    }

    public static Texture LoadLUTAsTexture(float4[] lut) {
        var texture = new Texture2D(lut.Length, 1, TextureFormat.RGFloat, false, true);
        texture.SetPixels(lut.Select(v => new Color(v.x, v.y, v.z, v.w)).ToArray());
        texture.Apply();
        return texture;
    }


    private static double DrainesPhaseFunction(double alpha, double g, double theta) {
        return 
        1.0 / (4.0 * Math.PI) * 
        (1 - g*g) / Math.Pow(1 + g*g - 2*g*Math.Cos(theta), 1.5) *
        (1 + alpha * Math.Pow(Math.Cos(theta), 2)) / (1 + alpha * (1 + 2*g*g)/3.0);
    }

    public static Texture CreateMieScatteringLUT() {
        Func<float,float> mieScatter = (float theta) =>
        {
            const float forward_bias = 0.3f; // Valid values are [-0.9,0.9] where negative values prioritize backscattering.
            const float softener = 0.5f; // Softens the distribution to be closer to uniform. 0 means nothing scatters perpendicular.
            const float lobe_sharpness = 2;

            // The model here is an artistic interpretation of something in between Rayleigh and Mie scattering.
            // It's a little silly to go for physical realism in this context, so instead this model goes for tweakability and "effect."
            var cos = Mathf.Cos(theta);

            return (softener + Mathf.Pow(cos, lobe_sharpness)) / (1 + forward_bias * cos);
        };

        var table = GenerateFunctionTable(mieScatter, -Mathf.PI, Mathf.PI);
        var normalizedTable = new float[table.Length];
        NormalizeDistribution(table, normalizedTable);
        var integral = new float[table.Length];
        IntegrateDistribution(normalizedTable, integral);
        var inverted = new float[table.Length];
        Invert(integral, -Mathf.PI, Mathf.PI, inverted, out float inverseStart, out float inverseEnd);
        var vectorTable = new float2[table.Length];
        AngleFunctionToVectorFunction(inverted, vectorTable);

        return LoadLUTAsTexture(vectorTable);
    }

    public static Texture CreateTeardropScatteringLUT() {
        Func<float,float> teardrop = (float theta) => {
            var x = theta / Mathf.PI - 1;
            return 0.5f + Mathf.Pow(x,6);
        };

        var table = GenerateFunctionTable(teardrop, 0, 2 * Mathf.PI, 65536);
        var normalizedTable = new float[table.Length];
        var avg = NormalizeDistribution(table, normalizedTable);
        var integral = new float[table.Length];
        IntegrateDistribution(normalizedTable, integral);
        var inverted = new float[2048];
        Invert(integral, 0, 2*Mathf.PI, inverted, out float inverseStart, out float inverseEnd);
        var vectorTable = new float2[inverted.Length];
        AngleFunctionToVectorFunction(inverted, vectorTable);
        var finalTable = new float3[inverted.Length];
        ComponentwiseGlue(vectorTable, inverted, finalTable);

        for(int i = 0;i < finalTable.Length;i++) {
            finalTable[i].z = avg / teardrop(finalTable[i].z);
        }

        return LoadLUTAsTexture(finalTable);
    }
}