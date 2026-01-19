#ifndef _SIMULATION_COMMON_
#define _SIMULATION_COMMON_

#include "PhotonerCommon.cginc"
#include "Random.cginc"

DECLARE_LUT(float2, g_mieScatteringLUT)
DECLARE_LUT(float3, g_teardropScatteringLUT)
DECLARE_LUT_3D(float4, g_bdrfLUT)

Texture2D<float4> g_albedo;
Texture2D<float4> g_transmissibility;
SamplerState samplerg_transmissibility;
Texture2D<float4> g_normalAlignment;
SamplerState samplerg_normalAlignment;
Texture2D<float4> g_quadTreeLeaves;

float2 g_importance_sampling_target;

struct Ray {
    float2 Origin;
    float2 Direction;
    float3 Energy;
};

struct IntegrationContext
{
    Ray photon;
    float uHitCurrent;
    float uHitNext;
    float uEscape;
    float2 testUV;
    float4 transmissibilityCurrent;
    float4 transmissibilityNext;
    int lod;

    void Init(Ray photon) {
        this.photon = photon;
        uHitCurrent = 0;
        uHitNext = 0;
        uEscape = 0;
        testUV = float2(0,0);
        transmissibilityCurrent = float4(1,1,0,0);
        transmissibilityNext = float4(1,1,0,0);
        lod = 0;
    }
};

interface IMonteCarloMethod
{
    // Called when a new raycast begins or after a bounce
    void BeginTraversal(inout IntegrationContext ctx);

    // Called to test for hit condition (overshoot)
    // When testing, the implementation is required to maintain any state necessary for propagation.
    // When Propagate is called, the implementation should update its state to reflect propagation over the tested segment.
    // Returns true upon hit/overshoot, false to continue propagation
    bool Test(inout IntegrationContext ctx);

    // Called to propagate state forward when no hit occurs
    // Returns true to continue propagation, false to abort
    bool Propagate(inout IntegrationContext ctx);

    // Called when raycast ends
    // Once completed, a traversal may alter the photon state in response to, for example, absorption.
    // Returns true if processing should continue, false to abort
    bool EndTraversal(inout IntegrationContext ctx);

    // Called after EndTraversal to adjust photon state for bounces
    bool Bounce(inout IntegrationContext ctx, float3 albedo);
};

struct BaseContext {
    Random rand;
    float2 targetSize;

    void Init_BaseContext(uint4 seed) {
        rand.Init(seed);
        
        // TODO: Shall we always assume the GBuffer and output buffer are the same resolution?
        targetSize = TextureSize(g_albedo);
    }

    void Init(uint4 seed) {
        Init_BaseContext(seed);
    }

    // void Init(Random r) {
    //     rand = r;
    // }

    float2 ScatterMie(float2 incomingDirection) {
        float2 perp = incomingDirection.yx;
        perp.x *= -1;
    
        float2 scatter = SampleLUT(g_mieScatteringLUT, rand.Next());
        return scatter.x * incomingDirection + scatter.y * perp;
    }
    
    float3 ScatterImportance(float2 origin) {
        float2 important_direction = g_importance_sampling_target - origin;
        float lsq = dot(important_direction, important_direction);
    
        if(false && lsq < 1/16.0) {
            return float3(rand.NextDirection(), 1);
        } else {
            important_direction /= -sqrt(lsq);
            float2 perp = float2(-important_direction.y, important_direction.x);
            float3 sample = SampleLUT(g_teardropScatteringLUT, rand.Next());
            return float3(important_direction * sample.x + perp * sample.y, sample.z);
        }
    }

    float4 CubicWeights(float u) {
        float4x4 basis = {
            {-0.5,  1.5, -1.5,  0.5},
            {   1, -2.5,    2, -0.5},
            {-0.5,    0,  0.5,    0},
            {   0,    1,    0,    0}
        };

        float uu = u*u;
        float4 series = {uu*u, uu, u, 1};
        return mul(series, basis);
    }

    float4 HermiteWeights(float u) {
        float4x4 basis = {
            { 2,  1, -2,  1},
            {-3, -2,  3, -1},
            { 0,  1,  0,  0},
            { 1,  0,  0,  0}
        };

        float uu = u*u;
        float4 series = {uu*u, uu, u, 1};
        return mul(series, basis);
    }

    float3 BilinearAsCubic(float u) {
        float4 weights = CubicWeights(u);
        float w = weights.x + weights.y;

        return float3(
            /*s = */ weights.x / w - 1 - u,
            /*t = */ weights.z / (1-w) + 1 - u,
            w
        );
    }

    float3 StandardBRDF(float2 normal, float2 reflected, float roughness) {
        float3 uvw = float3(
            rand.Next(),
            (cross2D(normal, reflected) + 1.0) / 2.0,
            roughness);
        float2 tangent = float2(-normal.y, normal.x);

        uint width, height, depth;
        g_bdrfLUT.GetDimensions(width, height, depth);

        float3 rescaled_uvw = {
            dot(float2(1,uvw.x),lut_window_g_bdrfLUT.xy),
            dot(float2(1,uvw.y),lut_window_g_bdrfLUT.zw),
            dot(float2(1,uvw.z),lut_slice_window_g_bdrfLUT)
        };

        float u_in_pixel_space = rescaled_uvw.x * width - 0.5;
        float bilinearParam = frac(u_in_pixel_space);
        float4 weights = CubicWeights(bilinearParam);
        float u_p1 = u_in_pixel_space - bilinearParam;
        float u_p2 = u_p1 + 1;

        float3 uvw_1 = rescaled_uvw;
        uvw_1.x = (u_p1 + 0.5) / width;
        float3 uvw_2 = rescaled_uvw;
        uvw_2.x = (u_p2 + 0.5) / width;

        float4 scattered_1 = g_bdrfLUT.SampleLevel(samplerg_bdrfLUT, uvw_1, 0);
        float4 scattered_2 = g_bdrfLUT.SampleLevel(samplerg_bdrfLUT, uvw_2, 0);

        float4 scattered;

        float4 tangent_1 = float4(-scattered_1.y, scattered_1.x, 0, 0) * scattered_1.z;
        float4 tangent_2 = float4(-scattered_2.y, scattered_2.x, 0, 0) * scattered_2.z;

        float4 hermiteWeights = HermiteWeights(bilinearParam);
        scattered = 
            scattered_1 * hermiteWeights.x +
            tangent_1 * hermiteWeights.y +
            scattered_2 * hermiteWeights.z +
            tangent_2 * hermiteWeights.w;

        //scattered = SampleLUT3D(g_bdrfLUT, uvw);

        return float3(normalize(scattered.x * normal + scattered.y * tangent), scattered.w*scattered.w);
    }
    
    float4 ScatterMaterially(inout float2 origin, float2 origin_uv, float2 incoming)
    {
        const float eps = 1e-5f;
        float4 normal_alignment = g_normalAlignment.SampleLevel(samplerg_normalAlignment, origin_uv, 0);
        float3 normal = normal_alignment.xyz;
        float alignment = normal_alignment.w;

        if(dot(normal.xy, normal.xy) < eps) {
            // No normal information, scatter uniformly
            float2 dir = rand.NextDirection();
            return float4(dir, 1, 0);
        // } else if(dot(normal.xy, normal.xy) < 0.99) {
        //      // Small normal implies we're near a flat surface - continue propagating
        //      return (1).xxxx;
        } else if(dot(normal.xy, incoming) > 0) {
            // Normal is the same general direction as incoming - transmit
            // TODO: Allow the transmission to not count against bounce count.
            // BUG: Setting transmit to 1 causes an infinite loop when photons
            // are emitted within the normal field
            return float4(incoming, 1, /*transmit*/ 1-1);
        } else {
            // TODO: Allow BDRF to operate on not-fully-horizontal normals
            float len = length(normal.xy);
            float2 normal2D = normal.xy / len;
            float2 reflected = reflect(incoming, normal2D);
            
            alignment = saturate(alignment / len);
            origin -= incoming * 2.5;
            if(alignment > 0.999) {
                return float4(reflected, 1, 0);
            } else if(alignment == 0) {
                float2 dir = rand.NextDirection();
                return float4(dot(dir, normal2D) > 0 ? dir : -dir, 1, 0);
            } else {
                float3 scattered = StandardBRDF(normal2D, reflected, 1 - alignment);
                return float4(scattered, 0);
            }
        }
    }

    float4 ScatterMaterially(inout float2 origin, float2 incoming)
    {
        return ScatterMaterially(origin, origin / TextureSize(g_normalAlignment), incoming);
    }
};


#endif // _SIMULATION_COMMON_