using System.Collections.Generic;
using UnityEngine;
using System.Linq;

using Random = System.Random;

public static class RandExtensions {
    public static float NextSingle(this Random rand) {
        return (float)rand.NextDouble();
    }

    public static float NextRange(this Random rand, float min, float max, float bias = 0) {
        return Mathf.Pow(rand.NextSingle(), Mathf.Pow(10, -bias)) * (max - min) + min;
    }

    public static bool NextBool(this Random rand, float weight = 0.5f) {
        return rand.NextSingle() < weight;
    }

    public static T NextWeightedOption<T>(this Random rand, Dictionary<T,float> weights) {
        var total = weights.Values.Sum();
        var val = rand.NextSingle() * total;

        foreach(var kvp in weights)
        {
            if(val <= kvp.Value)
                return kvp.Key;
            val -= kvp.Value;
        }
        return weights.Keys.Last();
    }

    public static Color NextLightColor(this Random rand) {
        return Color.HSVToRGB(rand.NextSingle(), Mathf.Sqrt(rand.NextSingle()), 1);
    }
}