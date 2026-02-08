#ifndef _SIMULATION_COMMON_
#define _SIMULATION_COMMON_

#include "LitboxCommon.cginc"
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

Texture2D<float> g_importanceMap;
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
    float2 pixelSize;

    void Init_BaseContext(uint4 seed) {
        rand.Init(seed);
        
        // TODO: Shall we always assume the GBuffer and output buffer are the same resolution?
        targetSize = TextureSize(g_albedo);
        pixelSize = 1.0f / targetSize;
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
    
    float3 ScatterImportanceLobed(float2 origin) {
        float2 important_direction = g_importance_sampling_target - origin;
        //float2 important_direction = float2(1,0);
        float lsq = dot(important_direction, important_direction);
    
        if(false && lsq < 1/16.0) {
            return float3(rand.NextDirection(), 1);
        } else {
            important_direction /= -sqrt(lsq);
            //important_direction = abs(important_direction);
            float2 perp = float2(-important_direction.y, important_direction.x);
            float3 sample = SampleLUT(g_teardropScatteringLUT, rand.Next());
            return float3(important_direction * sample.x + perp * sample.y, sample.z);
            //return float3(important_direction * sample.x + perp * sample.y, sample.z);
        }
    }

    float3 Test_Square_PDF() {
        float2 square = { rand.Next() - 0.5, rand.Next() - 0.5 };

        float picker = rand.Next();

        float4 odds = { 0.05, 0.8, 0.13, 0.02 };
        float weight = 1;

        if(picker < odds.x) {
            weight = odds.x;
            square = square / 2 + float2(-0.25, 0.25);
        } else if (picker < odds.x + odds.y) {
            weight = odds.y;
            square = square / 2 + float2(0.25, 0.25);
        } else if (picker < odds.x + odds.y + odds.z) {
            weight = odds.z;
            square = square / 2 + float2(-0.25, -0.25);
        } else {
            weight = odds.w;
            square = square / 2 + float2(0.25, -0.25);
        }

        return float3(square, 1.0 / weight);
    }

    float3 TestImportanceMapPDF(float2 origin_uv) {
        float2 uv = origin_uv;
        float randSelector = rand.Next();
        float2 pixelSize = 1.0f / TextureSize(g_importanceMap);
        float2 uvShift = 4 * pixelSize;
        float2 jitter = rand.Next2() - 0.5f;
        float totalEnergy = 1;
        float selectedEnergy = 0;

        // m0 shift distance: pixelSize / 2
        // m1: pixelSize
        // m2: pixelSize * 2
        // m3: pixelSize * 4

        [unroll]
        for (int m = 2; m >= 2; m--) {
            float4 weights = g_importanceMap.Gather(sampler_point_clamp, uv, m);
            
            // quadrants are returned in a specific order: 
            // W=Top-Left (-,+), Z=Top-Right (+,+), X=Bottom-Left (-,-), Y=Bottom-Right (+,-)
            float total = weights.x + weights.y + weights.z + weights.w;

            if(m == 3) {
                totalEnergy = total;
            }
            
            if (total <= 0) { break; }

            float4 p = weights / total;
            
            if (randSelector < p.w) {
                uv += float2(-uvShift.x, uvShift.y);
                selectedEnergy = p.w;
                randSelector /= p.w;
            } else if (randSelector < p.w + p.z) {
                uv += float2(uvShift.x, uvShift.y);
                selectedEnergy = p.z;
                randSelector = (randSelector - p.w) / p.z;
            } else if (randSelector < p.w + p.z + p.x) {
                uv += float2(-uvShift.x, -uvShift.y);
                selectedEnergy = p.x;
                randSelector = (randSelector - p.w - p.z) / p.x;
            } else {
                uv += float2(uvShift.x, -uvShift.y);
                selectedEnergy = p.y;
                randSelector = (randSelector - p.w - p.z - p.x) / p.y;
            }
            
            uvShift *= 0.5f;
        }

        float2 n = normalize(uv - origin_uv);
        float2 check = normalize(float2(1,1));
        return float3(uv - origin_uv + jitter * 4 * uvShift, 1.0 / selectedEnergy);
    }

    float3 ScatterImportanceGuided_Test(float2 origin_uv) {
        float3 squareWeighted = TestImportanceMapPDF(origin_uv);
        //float3 squareWeighted = Test_Square_PDF();
        float2 dir = normalize(squareWeighted.xy);

        // Properly weight the corners of the square to balance the projection onto a circle and still be a proper PDF.
        float2 c = abs(dir);
        float r = rsqrt((c.x + c.y) / max(c.x, c.y));
        return float3(dir, squareWeighted.z * r * r * r / (2 * PI));
    }

    float3 ScatterImportanceGuided(float2 origin_uv) {
        float2 uv = origin_uv;
        float randSelector = rand.Next();
        float2 pixelSize = 1.0f / TextureSize(g_importanceMap);
        float2 uvShift = 4 * pixelSize;
        float2 jitter = rand.Next2() - 0.5f;

        // m0 shift distance: pixelSize / 2
        // m1: pixelSize
        // m2: pixelSize * 2
        // m3: pixelSize * 4

        [unroll]
        for (int m = 3; m >= 0; m--) {
            float4 weights = g_importanceMap.Gather(sampler_point_clamp, uv, m);
            
            // quadrants are returned in a specific order: 
            // W=Top-Left (-,+), Z=Top-Right (+,+), X=Bottom-Left (-,-), Y=Bottom-Right (+,-)
            float total = weights.x + weights.y + weights.z + weights.w;
            
            if (total <= 0) { break; }

            float4 p = weights / total;
            
            if (randSelector < p.w) {
                uv += float2(-uvShift.x, uvShift.y);
                randSelector /= p.w;
            } else if (randSelector < p.w + p.z) {
                uv += float2(uvShift.x, uvShift.y);
                randSelector = (randSelector - p.w) / p.z;
            } else if (randSelector < p.w + p.z + p.x) {
                uv += float2(-uvShift.x, -uvShift.y);
                randSelector = (randSelector - p.w - p.z) / p.x;
            } else {
                uv += float2(uvShift.x, -uvShift.y);
                randSelector = (randSelector - p.w - p.z - p.x) / p.y;
            }
            
            uvShift *= 0.5f;
        }

        return 0;
        //return saturate(uv);
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
        return ScatterMaterially(origin, (origin + 0.5) / TextureSize(g_normalAlignment), incoming);
    }
};

void Integrate(inout Ray photon, uint bounces, IMonteCarloMethod state) {
    IntegrationContext ctx;
    ctx.Init(photon);

    float2 target_size = TextureSize(g_albedo);
    float2 pixel_size = 1.0f / target_size;

    for(uint bounce = 0;bounce < bounces;) {
        float uScatter;

        if(ctx.photon.Direction.x == 0) ctx.photon.Direction.x = 1e-8;
        if(ctx.photon.Direction.y == 0) ctx.photon.Direction.y = 1e-8;
    
        float2 uvOrigin = ctx.photon.Origin / target_size;
        float2 uvDirection = ctx.photon.Direction / target_size;
        float4 uBoundaryBox = (float4(-pixel_size,1 + pixel_size) - uvOrigin.xyxy) / uvDirection.xyxy;

        ctx.uEscape = min(max(uBoundaryBox[0], uBoundaryBox[2]), max(uBoundaryBox[1], uBoundaryBox[3]));
        ctx.uHitCurrent = 0;
        ctx.testUV = uvOrigin;

        state.BeginTraversal(ctx);
        bool continueRunning = true;
        for(int steps = 0;steps < 2000;steps++) {
            bool overshoot = false;

            // TODO: Reassess the utility of the quadtree. Tests showed no performance improvement using it.
            ctx.lod = 0;//g_quadTreeLeaves.SampleLevel(sampler_point_clamp, ctx.testUV, 0).x;

            //do {
                ctx.transmissibilityNext = g_transmissibility.SampleLevel(samplerg_transmissibility, ctx.testUV, ctx.lod);
                ctx.uHitNext = ctx.uHitCurrent + (1 << ctx.lod);
                overshoot = state.Test(ctx);
                
            //     if (!overshoot) { break; }
            //     if (ctx.lod <= 0) { break; }
            //     ctx.lod--;
            // } while (true);
        
            if(!overshoot) { // Keep propagating
                ctx.uHitCurrent = ctx.uHitNext;
                ctx.testUV = uvOrigin + uvDirection * ctx.uHitCurrent;
                if(!state.Propagate(ctx)) {
                    continueRunning = false;
                    break;
                }
            } else { // Scatter occurs within this sample, end traversal
                continueRunning = state.EndTraversal(ctx);
                break;
            }
        }

        // if(failure) {
        //     // Fail condition... traversal took too many steps
        //     WritePhoton(ctx.photon.Origin, uint3(1000000000, 1000000000, 0));
        //     WritePhoton(ctx.testUV * g_target_size, uint3(0, 1000000000, 0));
        //     return false;,
        // } 

        if(!continueRunning) break;

        ctx.photon.Origin += ctx.photon.Direction * ctx.uHitCurrent;
        float3 albedo = g_albedo.SampleLevel(sampler_point_clamp, ctx.testUV, 0).rgb;
        if(state.Bounce(ctx, albedo)) {
            bounce++;
        }
    }

    photon = ctx.photon;
}

#endif // _SIMULATION_COMMON_