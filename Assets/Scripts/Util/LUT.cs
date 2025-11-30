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
            float y = inverseStart + i * (inverseEnd - inverseStart) / (outInverse.Length - 1);

            int xLow = function.Length - 1;

            for(int k = 0;k < function.Length;k++)
            {
                if(function[k] > y) {
                    xLow = k-1;
                    break;
                }
            }

            if(xLow == function.Length - 1) {
                outInverse[i] = domainStart + xLow * (domainEnd - domainStart) / (function.Length - 1);
                continue;
            }

            // Binary search to drill down to a precise x value
            float L = xLow;
            float H = xLow + 1;

            float yL = function.ReadCubic(L);
            float yH = function.ReadCubic(H);

            while(H-L > 1e-5) {
                // if(yL > y || yH < y) {
                //     throw new Exception("oops");
                // }

                float M = (L+H)/2;
                float yM = function.ReadCubic(M);

                if(yM < y) {
                    if(L == M) {
                        break;
                    }
                    L = M;
                    yL = yM;
                } else {
                    if(H == M) {
                        break;
                    }
                    H = M;
                    yH = yM;
                }
            }

            float x = domainStart + L * (domainEnd - domainStart) / (function.Length - 1);

            outInverse[i] = x;
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

    public static void ComponentwiseGlue(float[] x, float[] y, float2[] outGlued) {
        for(int i = 0;i < outGlued.Length;i++) {
            outGlued[i] = new float2(x[i], y[i]);
        }
    }

    /// <summary>
    /// Creates a random angle generator table fitting a probability distribution function
    /// </summary>
    /// <param name="relativePDF">
    /// A function that will be sampled from -PI to PI for angle probabilities.
    /// The function does not need to be normalized, but it must be nonnegative across the domain [-PI,PI].
    /// </param>
    /// <remarks>
    /// To use the generator, generate a random number from 0-1 and use it to sample
    /// the table as you would a texture on the GPU.
    /// When sampling on the GPU, it is important to adjust the sampling coordinate
    /// so that 0 lands on the center of the first texel, and 1 lands on the center of
    /// the last texel. This can be achieved with the following formula,
    /// given the size of the table N:
    ///
    /// uAdjusted = 0.5/N + u * (1 - 1/N)
    ///
    /// </remarks>
    /// <returns>
    /// An array that can be loaded as a texture into the GPU.
    /// The XY components of each value represent a unit-length vector
    /// pointing in the direction of the output angle. The Z component
    /// represents the inverse of the density at the output angle.
    /// </returns>
    public static float3[] CreateVectorizedAnglePDFLUT(Func<float,float> relativePDF, int samples = 2048, float lower = -Mathf.PI, float upper = Mathf.PI) {
        var table = GenerateFunctionTable(relativePDF, lower, upper, samples);
        var normalizedTable = new float[table.Length];
        var avg = NormalizeDistribution(table, normalizedTable);
        var integral = new float[table.Length];
        IntegrateDistribution(normalizedTable, integral);
        var inverted = new float[table.Length];
        Invert(integral, lower, upper, inverted, out float inverseStart, out float inverseEnd);
        var vectorTable = new float2[table.Length];
        AngleFunctionToVectorFunction(inverted, vectorTable);
        var finalTable = new float3[inverted.Length];
        ComponentwiseGlue(vectorTable, inverted, finalTable);

        for(int i = 0;i < finalTable.Length;i++) {
            finalTable[i].z = avg / relativePDF(finalTable[i].z);
        }

        return finalTable;
    }

    /// <summary>
    /// Creates a random generator table fitting a probability distribution function
    /// </summary>
    /// <param name="relativePDF">
    /// A function that will be sampled for probabilities.
    /// The function does not need to be normalized, but it must be nonnegative across the domain [lowerbound,upperbound].
    /// </param>
    /// <remarks>
    /// To use the generator, generate a random number from 0-1 and use it to sample
    /// the table as you would a texture on the GPU.
    /// When sampling on the GPU, it is important to adjust the sampling coordinate
    /// so that 0 lands on the center of the first texel, and 1 lands on the center of
    /// the last texel. This can be achieved with the following formula,
    /// given the size of the table N:
    ///
    /// uAdjusted = 0.5/N + u * (1 - 1/N)
    ///
    /// </remarks>
    /// <returns>
    /// An array that can be loaded as a texture into the GPU.
    /// The X component of each value represents the distributed output,
    /// The Y component represents the inverse of the density at that value.
    /// </returns>
    public static float2[] CreatePDFLUT(Func<float,float> relativePDF, float lowerbound = 0, float upperbound = 1, int samples = 2048) {
        var table = GenerateFunctionTable(relativePDF, lowerbound, upperbound, samples);
        var normalizedTable = new float[table.Length];
        var avg = NormalizeDistribution(table, normalizedTable);
        var integral = new float[table.Length];
        IntegrateDistribution(normalizedTable, integral);
        var inverted = new float[table.Length];
        Invert(integral, lowerbound, upperbound, inverted, out float inverseStart, out float inverseEnd);
        var finalTable = new float2[inverted.Length];
        ComponentwiseGlue(inverted, inverted, finalTable);

        for(int i = 0;i < finalTable.Length;i++) {
            finalTable[i].y = avg / relativePDF(finalTable[i].y);
        }

        return finalTable;
    }

    public static float3[] CreateMieScatteringLUT() =>
        CreateVectorizedAnglePDFLUT((float theta) => {
            const float forward_bias = 0.3f; // Valid values are [-0.9,0.9] where negative values prioritize backscattering.
            const float softener = 0.5f; // Softens the distribution to be closer to uniform. 0 means nothing scatters perpendicular.
            const float lobe_sharpness = 2;

            // The model here is an artistic interpretation of something in between Rayleigh and Mie scattering.
            // It's a little silly to go for physical realism in this context, so instead this model goes for tweakability and "effect."
            var cos = Mathf.Cos(theta);

            return (softener + Mathf.Pow(cos, lobe_sharpness)) / (1 + forward_bias * cos);
        });

    public static float3[] CreateTeardropScatteringLUT(float spikeStrength) =>
        CreateVectorizedAnglePDFLUT((float theta) => {
            var x = theta / Mathf.PI;
            return 1 + spikeStrength * Mathf.Pow(x,6);
        });

    // TODO: Probably remove this. We have better solutions now.
    public static float2[,,] CreateQuantumTunnelingLUT() {
        // Quantum tunneling LUT is 3 dimensional.
        // The 3 dimensions are:
        // x - scatter potential (PDF dependent)
        // y - focal distance (in terms of scatter potential)
        // z - sample spread [0.00001,0.99]. Equivalent to net transmissibility over the standard deviation:  T^d

        Func<float,float,float,float> tunnelingPDF = (float x, float focalDistance, float spread) =>
            Mathf.Exp(-Mathf.Pow(Mathf.Log(focalDistance/x) / Mathf.Log(spread),2));

        //float2[,,] output = new float2[2048,256,16];
        float2[,,] output = new float2[512,16,16];

        for(int k = 0;k < output.GetLength(2);k++) {
            float spread = 0.00001f + (0.99f - 0.00001f) * k / output.GetLength(2);
            for(int j = 0;j < output.GetLength(1);j++) {
                float focalDistance = (float)j / output.GetLength(1);
                float2[] linePDF = CreatePDFLUT(x => tunnelingPDF(x, focalDistance, spread), 1e-4f, 1, output.GetLength(0));

                for(int i = 0;i < linePDF.GetLength(0);i++) {
                    output[i,j,k] = linePDF[i];
                }
            }
        }

        return output;
    }

    public static float4[,,] CreateBDRFLUT() {
        // BDRF LUT is 3 dimensional.
        // x - random scatter (PDF)
        // y - (cross2D(normal, reflected) + 1.0) / 2.0
        // z - roughness

        float4[,,] output = new float4[63,64,16];

        for(int j = 0;j < output.GetLength(1);j++) {
            float v = (float)j / (output.GetLength(1) - 1);
            float normalCrossIncident = 2 * v - 1;
            float incidentAngle = Mathf.Asin(normalCrossIncident);
            for(int k = 0;k < output.GetLength(2);k++) {
                float roughness = (float)k / (output.GetLength(2) - 1);

                if(roughness == 0) {
                    float4 reflected = new float4(Mathf.Cos(-incidentAngle), Mathf.Sin(-incidentAngle), 1, 1);
                    for(int i = 0;i < output.GetLength(0);i++) {
                        output[i,j,k] = reflected;
                    }
                } else {
                    float3[] linePDF = CreateVectorizedAnglePDFLUT((float theta) => {
                        // Use Trowbridge-Reitz (GGX) NDF as the basis for our BDRF. It looks nice.
                        var halfAngle = (theta + incidentAngle) / 2;
                        var R = roughness * roughness;

                        var cosHalf = Mathf.Cos(halfAngle);
                        return 1.0f/Mathf.Pow(cosHalf*cosHalf*(R*R-1) + 1,2);
                    }, output.GetLength(0), -Mathf.PI/2+0.0001f, Mathf.PI/2-0.0001f);

                    // BDRFs with narrow scattering lobes can have zero probability at the edges.
                    // Detect these and adjust to avoid artifacts.
                    if(linePDF[0].z > 10)
                    {
                        var angle1 = Mathf.Acos(linePDF[1].y);
                        var angle2 = Mathf.Acos(linePDF[2].y);
                        var angle0 = angle1 - (angle2 - angle1);
                        var vector0 = new float2(Mathf.Sin(angle0), Mathf.Cos(angle0));
                        linePDF[0] = new float3(vector0, linePDF[1].z);
                    }

                    var last = linePDF.Length - 1;
                    if(linePDF[last].z > 10)
                    {
                        var angle1 = Mathf.Acos(linePDF[last-1].y);
                        var angle2 = Mathf.Acos(linePDF[last-2].y);
                        var angle0 = angle1 + (angle1 - angle2);
                        var vector0 = new float2(Mathf.Sin(angle0), Mathf.Cos(angle0));
                        linePDF[last] = new float3(vector0, linePDF[last-1].z);
                    }
                   
                    for(int i = 0;i < linePDF.GetLength(0);i++) {
                        if(i == 0 || i == linePDF.Length-1) {
                            output[i,j,k] = new float4(linePDF[i].x, linePDF[i].y, linePDF[i].z, 0);
                            //output[i,j,k] = new float4(0, 0, linePDF[i].z, 0);
                        } else {
                            output[i,j,k] = new float4(linePDF[i].x, linePDF[i].y, linePDF[i].z, 1);
                        }
                    }
                }
            }
        }

        return output;
    }
}