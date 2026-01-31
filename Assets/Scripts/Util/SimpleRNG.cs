using System;
using Unity.Mathematics;
using UnityEngine;

/// <summary>
/// SimpleRNG is a simple random number generator based on 
/// George Marsaglia's MWC (multiply with carry) generator.
/// Although it is very simple, it passes Marsaglia's DIEHARD
/// series of random number generator tests.
/// 
/// Written by John D. Cook 
/// http://www.johndcook.com
/// </summary>
public struct SimpleRNG
{
    const uint DefaultW = 521288629;
    const uint DefaultZ = 362436069;

    uint _w;
    uint _z;

    public SimpleRNG(uint seed)
    {
        // These values are not magical, just the default values Marsaglia used.
        // Any pair of unsigned integers should be fine.
        _w = DefaultW;
        _z = DefaultZ;

        SetSeed(seed);
    }

    // The random generator seed can be set three ways:
    // 1) specifying two non-zero unsigned integers
    // 2) specifying one non-zero unsigned integer and taking a default value for the second
    // 3) setting the seed from the system time

    public void SetSeed(uint u, uint v = 0)
    {
        _w = (u == 0) ? DefaultW : u;
        _z = (v == 0) ? DefaultZ : v;
    }

    public void SetSeedFromSystemTime()
    {
        System.DateTime dt = System.DateTime.Now;
        long x = dt.ToFileTime();
        SetSeed((uint)(x >> 16), (uint)(x % 4294967296));
    }

    // This is the heart of the generator.
    // It uses George Marsaglia's MWC algorithm to produce an unsigned integer.
    // See http://www.bobwheeler.com/statistics/Password/MarsagliaPost.txt
    public uint NextUint()
    {
        _z = 36969 * (_z & 65535) + (_z >> 16);
        _w = 18000 * (_w & 65535) + (_w >> 16);
        return (_z << 16) + _w;
    }

    // Produce a uniform random sample from the open interval (0, 1).
    // The method will not return either end point.
    public double NextDouble()
    {
        // 0 <= u < 2^32
        uint u = NextUint();
        // The magic number below is 1/(2^32 + 2).
        // The result is strictly between 0 and 1.
        return (u + 1.0) * 2.328306435454494e-10;
    }

    public float NextFloat()
    {
        return (float)NextDouble();
    }

    public float2 NextFloat2()
    {
        return new float2(NextFloat(), NextFloat());
    }

    public float3 NextFloat3()
    {
        return new float3(NextFloat(), NextFloat(), NextFloat());
    }

    public float4 NextFloat4()
    {
        return new float4(NextFloat(), NextFloat(), NextFloat(), NextFloat());
    }

    #region NextNormal

    // Get normal (Gaussian) random sample with mean 0 and standard deviation 1
    public double NextNormal()
    {
        // Use Box-Muller algorithm
        double u1 = NextDouble();
        double u2 = NextDouble();
        double r = Math.Sqrt(-2.0 * Math.Log(u1));
        double theta = 2.0 * Math.PI * u2;
        return r * Math.Sin(theta);
    }

    public float NextNormal1F()
    {
        // Use Box-Muller algorithm
        var u1 = NextFloat();
        var u2 = NextFloat();
        var r = Mathf.Sqrt(-2.0f * Mathf.Log(u1));
        var theta = 2.0f * Mathf.PI * u2;
        return r * Mathf.Sin(theta);
    }

    public float2 NextNormal2F()
    {
        // Use Box-Muller algorithm
        var u1 = NextFloat2();
        var u2 = NextFloat2();
        var r = math.sqrt(-2.0f * math.log(u1));
        var theta = 2.0f * Mathf.PI * u2;
        return r * math.sin(theta);
    }

    public float3 NextNormal3F()
    {
        // Use Box-Muller algorithm
        var u1 = NextFloat3();
        var u2 = NextFloat3();
        var r = math.sqrt(-2.0f * math.log(u1));
        var theta = 2.0f * Mathf.PI * u2;
        return r * math.sin(theta);
    }

    public float4 NextNormal4F()
    {
        // Use Box-Muller algorithm
        var u1 = NextFloat4();
        var u2 = NextFloat4();
        var r = math.sqrt(-2.0f * math.log(u1));
        var theta = 2.0f * Mathf.PI * u2;
        return r * math.sin(theta);
    }

    #endregion

    // Get normal (Gaussian) random sample with specified mean and standard deviation
    public double NextNormal(double mean, double standardDeviation)
    {
        if (standardDeviation <= 0.0)
        {
            string msg = string.Format("Shape must be positive. Received {0}.", standardDeviation);
            throw new ArgumentOutOfRangeException(msg);
        }

        return mean + standardDeviation * NextNormal();
    }

    #region NextExponential

    // Get exponential random sample with mean 1
    public double NextExponential()
    {
        return -Math.Log(NextDouble());
    }

    public float NextExponential1F()
    {
        return -Mathf.Log(NextFloat());
    }

    public float2 NextExponential2F()
    {
        return -math.log(NextFloat2());
    }

    public float3 NextExponential3F()
    {
        return -math.log(NextFloat3());
    }

    public float4 NextExponential4F()
    {
        return -math.log(NextFloat4());
    }

    #endregion

    // Get exponential random sample with specified mean
    public double NextExponential(double mean)
    {
        if (mean <= 0.0)
        {
            string msg = string.Format("Mean must be positive. Received {0}.", mean);
            throw new ArgumentOutOfRangeException(msg);
        }

        return mean * NextExponential();
    }

    public double NextGamma(double shape, double scale)
    {
        // Implementation based on "A Simple Method for Generating Gamma Variables"
        // by George Marsaglia and Wai Wan Tsang.  ACM Transactions on Mathematical Software
        // Vol 26, No 3, September 2000, pages 363-372.

        double d, c, x, xsquared, v, u;

        if (shape >= 1.0)
        {
            d = shape - 1.0 / 3.0;
            c = 1.0 / Math.Sqrt(9.0 * d);
            for (; ; )
            {
                do
                {
                    x = NextNormal();
                    v = 1.0 + c * x;
                }
                while (v <= 0.0);
                v = v * v * v;
                u = NextDouble();
                xsquared = x * x;
                if (u < 1.0 - .0331 * xsquared * xsquared || Math.Log(u) < 0.5 * xsquared + d * (1.0 - v + Math.Log(v)))
                    return scale * d * v;
            }
        }
        else if (shape <= 0.0)
        {
            string msg = string.Format("Shape must be positive. Received {0}.", shape);
            throw new ArgumentOutOfRangeException(msg);
        }
        else
        {
            double g = NextGamma(shape + 1.0, 1.0);
            double w = NextDouble();
            return scale * g * Math.Pow(w, 1.0 / shape);
        }
    }

    public double NextChiSquare(double degreesOfFreedom)
    {
        // A chi squared distribution with n degrees of freedom
        // is a gamma distribution with shape n/2 and scale 2.
        return NextGamma(0.5 * degreesOfFreedom, 2.0);
    }

    public double NextInverseGamma(double shape, double scale)
    {
        // If X is gamma(shape, scale) then
        // 1/Y is inverse gamma(shape, 1/scale)
        return 1.0 / NextGamma(shape, 1.0 / scale);
    }

    public double NextWeibull(double shape, double scale)
    {
        if (shape <= 0.0 || scale <= 0.0)
        {
            string msg = string.Format("Shape and scale parameters must be positive. Recieved shape {0} and scale{1}.", shape, scale);
            throw new ArgumentOutOfRangeException(msg);
        }
        return scale * Math.Pow(-Math.Log(NextDouble()), 1.0 / shape);
    }

    public double NextCauchy(double median, double scale)
    {
        if (scale <= 0)
        {
            string msg = string.Format("Scale must be positive. Received {0}.", scale);
            throw new ArgumentException(msg);
        }

        double p = NextDouble();

        // Apply inverse of the Cauchy distribution function to a uniform
        return median + scale * Math.Tan(Math.PI * (p - 0.5));
    }

    public double NextStudentT(double degreesOfFreedom)
    {
        if (degreesOfFreedom <= 0)
        {
            string msg = string.Format("Degrees of freedom must be positive. Received {0}.", degreesOfFreedom);
            throw new ArgumentException(msg);
        }

        // See Seminumerical Algorithms by Knuth
        double y1 = NextNormal();
        double y2 = NextChiSquare(degreesOfFreedom);
        return y1 / Math.Sqrt(y2 / degreesOfFreedom);
    }

    // The Laplace distribution is also known as the double exponential distribution.
    public double NextLaplace(double mean, double scale)
    {
        double u = NextDouble();
        return (u < 0.5) ?
            mean + scale * Math.Log(2.0 * u) :
            mean - scale * Math.Log(2 * (1 - u));
    }

    public double NextLogNormal(double mu, double sigma)
    {
        return Math.Exp(NextNormal(mu, sigma));
    }

    public double NextBeta(double a, double b)
    {
        if (a <= 0.0 || b <= 0.0)
        {
            string msg = string.Format("Beta parameters must be positive. Received {0} and {1}.", a, b);
            throw new ArgumentOutOfRangeException(msg);
        }

        // There are more efficient methods for generating beta samples.
        // However such methods are a little more efficient and much more complicated.
        // For an explanation of why the following method works, see
        // http://www.johndcook.com/distribution_chart.html#gamma_beta

        double u = NextGamma(a, 1.0);
        double v = NextGamma(b, 1.0);
        return u / (u + v);
    }
}
