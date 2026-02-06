#ifndef _LEGACY_INTEGRATORS_
#define _LEGACY_INTEGRATORS_

#include "../Shaders/LitboxCommon.cginc"
#include "../Shaders/Random.cginc"
#include "../Shaders/SimulationCommon.cginc"

struct ImplicitIntegrator : BaseContext, IMonteCarloMethod
{
    float probability;
    float hitIntensity;
    float uTarget;
    float uEscape;
    float transmissibility;

    float tested_transmissibility;
    float tested_u;
        
    void Init(BaseContext ctx)
    {
        rand = ctx.rand;

        probability = 1;
        hitIntensity = 0;
        uTarget = 0;
        transmissibility = 1;
    }

    void BeginTraversal(inout IntegrationContext ctx)
    {
        this.uEscape = ctx.uEscape;
        uTarget = rand.Next() * ctx.uEscape; // TODO: Find ways to bias this towards important areas
        transmissibility = 1;
    }
    
    bool Test(inout IntegrationContext ctx)
    {
        tested_u = ctx.uHitNext;
        tested_transmissibility = ctx.transmissibilityNext.x;

        return ctx.uHitNext > uTarget;
    }
    
    bool Propagate(inout IntegrationContext ctx)
    {
        transmissibility *= tested_transmissibility;

        // If the density of the substrate is too high, consider this the new endpoint.
        //THIS CONDITION WILL BREAK REFLECTION WHEN DENSITY APPROACHES 1
        return transmissibility > 1e-9;
    }
    
    bool EndTraversal(inout IntegrationContext ctx)
    {
        ctx.uHitCurrent = uTarget;
        probability *= transmissibility * uEscape / 256.0; // Downscale to prevent blowout
        hitIntensity = probability * (1 - tested_transmissibility);
        return probability > 1e-9; 
    }
    
    bool Bounce(inout IntegrationContext ctx, float3 albedo)
    {
        ctx.photon.Energy *= albedo;

        //float3 important_direction = ScatterImportanceLobed(ctx.photon.Origin);
        float4 important_direction = ScatterMaterially(ctx.photon.Origin, ctx.photon.Direction);
        ctx.photon.Direction = important_direction.xy;
        ctx.photon.Energy *= important_direction.z;

        float outScatterDensity  = hitIntensity * important_direction.w;

        WritePhoton(ctx.photon.Origin, ctx.photon.Energy, outScatterDensity, false);//bounce == 0);
       // ctx.photon.Energy -= outScatter;
        return true;
    }
};

struct ImplicitIntervalIntegrator : BaseContext, IMonteCarloMethod
{
    float currentSample;
    float probability;
    float uSampleTarget;
    float uBounceTarget;
    float transmissibility;
    float bounceTransmissibility;

    float tested_transmissibility;
    float tested_u;
    void WriteSample(inout Ray photon, float u, int lod)
    {
        float2 pSample = photon.Origin + photon.Direction * u;
        float3 albedo = g_albedo.SampleLevel(sampler_point_clamp, pSample / g_target_size, 0).rgb;
        float3 outScatterDensity = probability * transmissibility * (1 - pow(tested_transmissibility, 1.0/(1 << lod)));
        WritePhoton(pSample, photon.Energy * albedo, outScatterDensity, false);//bounce == 0);
    }

    void Init(BaseContext ctx)
    {
        rand = ctx.rand;

        probability = 1;
        uBounceTarget = 0;
        transmissibility = 1;
    }

    void BeginTraversal(inout IntegrationContext ctx)
    {
        currentSample = 0;
        uSampleTarget = rand.Next() * g_integration_interval;
        uBounceTarget = rand.Next() * ctx.uEscape; // TODO: Find ways to bias this towards important areas
        transmissibility = 1;
    }
    
    bool Test(inout IntegrationContext ctx)
    {
        tested_u = ctx.uHitNext;
        tested_transmissibility = ctx.transmissibilityNext.x;

        return tested_u > ctx.uEscape;
    }
    
    bool Propagate(inout IntegrationContext ctx)
    {
        transmissibility *= tested_transmissibility;

        if(tested_u > uSampleTarget) {
            currentSample += 1;
            WriteSample(ctx.photon, uSampleTarget, ctx.lod);
            uSampleTarget = (currentSample + rand.Next()) * g_integration_interval;
        }

        if(ctx.uHitCurrent < uBounceTarget) {
            bounceTransmissibility = transmissibility;
        } 

        // If the density of the substrate is too high, consider this the new endpoint.
        //THIS CONDITION WILL BREAK REFLECTION WHEN DENSITY APPROACHES 1
        return transmissibility > 1e-9;
    }
    
    bool EndTraversal(inout IntegrationContext ctx)
    {
        ctx.uHitCurrent = uBounceTarget;
        transmissibility = bounceTransmissibility;
        probability *= transmissibility;// * ctx.uEscape / 256.0; // Downscale to prevent blowout
        return probability > 1e-7;
    }

    bool Bounce(inout IntegrationContext ctx, float3 albedo)
    {
        ctx.photon.Energy *= albedo;

        //float3 important_direction = ScatterImportanceLobed(photon.Origin);
        float4 important_direction = ScatterMaterially(ctx.photon.Origin, ctx.photon.Direction);
        ctx.photon.Direction = important_direction.xy;
        ctx.photon.Energy *= important_direction.z * ctx.uEscape / 256.0;
        return true;
    }
};

struct ExplicitIntegrator : BaseContext, IMonteCarloMethod
{
    float transmissibility;
    float transmitPotential;
    float quantumScale;
    float uEscape;
    
    float4 tested_transmissibility;
    float tested_u;

    void Init(BaseContext ctx)
    {
        rand = ctx.rand;
    }
    
    void BeginTraversal(inout IntegrationContext ctx) 
    {
        this.uEscape = ctx.uEscape;
        transmissibility = 1;

        float u = rand.Next();
        transmitPotential = u * u * u;
        quantumScale = 3 * u * u;
    }

    bool Test(inout IntegrationContext ctx)
    {
        tested_u = ctx.uHitNext;
        tested_transmissibility = ctx.transmissibilityNext;

        float minimumTransmissibility = ctx.transmissibilityNext.y;
        return minimumTransmissibility * transmissibility < transmitPotential;
    }

    bool Propagate(inout IntegrationContext ctx)
    {
        transmissibility *= tested_transmissibility.x;
        return true; //ctx.transmissibilityNext.x < ctx.uEscape;
    }

    bool EndTraversal(inout IntegrationContext ctx)
    {
        ctx.uHitCurrent = tested_u + log2(transmitPotential / transmissibility) / log2(tested_transmissibility.x);
        return true;
    }

    bool Bounce(inout IntegrationContext ctx, float3 albedo)
    {
        ctx.photon.Energy *= albedo * quantumScale;

        //float3 important_direction = ScatterImportanceLobed(photon.Origin);
        float4 important_direction = ScatterMaterially(ctx.photon.Origin, ctx.photon.Direction);
        ctx.photon.Direction = important_direction.xy;
        ctx.photon.Energy *= important_direction.z;

        float outScatterDensity = (1 - tested_transmissibility.x) * important_direction.w;

        WritePhoton(ctx.photon.Origin, ctx.photon.Energy, outScatterDensity, false);//bounce == 0);
        //ctx.photon.Energy -= outScatter;
        return true;
    }
};

struct ExplicitBoundedIntegrator : BaseContext, IMonteCarloMethod
{
    bool searchingPhase;
    float transmissibility;
    float transmitPotential;
    float quantumScale;
    float uEscape;
    
    float4 tested_transmissibility;
    float tested_u;

    void Init(BaseContext ctx)
    {
        rand = ctx.rand;
        searchingPhase = true;
    }
    
    void BeginTraversal(inout IntegrationContext ctx) 
    {
        this.uEscape = ctx.uEscape;
        transmissibility = 1;
    }

    bool Test(inout IntegrationContext ctx)
    {
        tested_u = ctx.uHitNext;
        tested_transmissibility = ctx.transmissibilityNext;

        if(searchingPhase) {
            return tested_u > ctx.uEscape; 
        } else {
            float minimumTransmissibility = ctx.transmissibilityNext.y;
            return minimumTransmissibility * transmissibility < transmitPotential;
        }
    }

    bool Propagate(inout IntegrationContext ctx)
    {
        transmissibility *= tested_transmissibility.x;
        return true; //ctx.transmissibilityNext.x < ctx.uEscape;
    }

    bool EndTraversal(inout IntegrationContext ctx)
    {
        if(!searchingPhase) {
            ctx.uHitCurrent = tested_u + log2(transmitPotential / transmissibility) / log2(tested_transmissibility.x);
        } else {
            ctx.uHitCurrent = 0;
        }
        return true;
    }

    bool Bounce(inout IntegrationContext ctx, float3 albedo)
    {
        if(!searchingPhase) {
            ctx.photon.Energy *= albedo * quantumScale;

            //float3 important_direction = ScatterImportanceLobed(photon.Origin);
            float4 important_direction = ScatterMaterially(ctx.photon.Origin, ctx.photon.Direction);
            ctx.photon.Direction = important_direction.xy;
            ctx.photon.Energy *= important_direction.z;

            float outScatterDensity = (1 - tested_transmissibility.x) * important_direction.w;

            WritePhoton(ctx.photon.Origin, ctx.photon.Energy, outScatterDensity, false);//bounce == 0);
        } else {
            float u = rand.Next(transmissibility, 1);
            transmitPotential = u;
            quantumScale = 1;
            ctx.photon.Energy *= (1 - transmissibility);
            // transmitPotential = u * u * u;
            // quantumScale = 3 * u * u;
        }
        searchingPhase = !searchingPhase;
        return searchingPhase;
    }
};

struct ExplicitBounceImplicitInvervalIntegrator : BaseContext, IMonteCarloMethod
{
    float transmitPotential;
    float quantumScale;
    float currentSample;
    float hitIntensity;
    float uSampleTarget;
    float transmissibility;

    float tested_transmissibility;
    float tested_u;
    void WriteSample(inout Ray photon, float u, int lod)
    {
        float2 pSample = photon.Origin + photon.Direction * u;
        float3 albedo = g_albedo.SampleLevel(sampler_point_clamp, pSample / g_target_size, 0).rgb;
        float outScatterDensity = transmissibility * (1 - pow(tested_transmissibility, 1.0/(1 << lod)));
        WritePhoton(pSample, photon.Energy * albedo, outScatterDensity, false);//bounce == 0);
        photon.Energy -= photon.Energy * albedo * outScatterDensity;
    }

    void Init(BaseContext ctx)
    {
        rand = ctx.rand;

        hitIntensity = 0;
        transmissibility = 1;

        float u = rand.Next();
        transmitPotential = u * u * u;
        quantumScale = 3 * u * u;
    }

    void BeginTraversal(inout IntegrationContext ctx)
    {
        currentSample = 0;
        uSampleTarget = rand.Next() * g_integration_interval;
        transmissibility = 1;
    }
    
    bool Test(inout IntegrationContext ctx)
    {
        tested_u = ctx.uHitNext;
        tested_transmissibility = ctx.transmissibilityNext.x;

        float minimumTransmissibility = ctx.transmissibilityNext.y;
        return ctx.uHitNext > ctx.uEscape || minimumTransmissibility * transmissibility < transmitPotential;
    }
    
    bool Propagate(inout IntegrationContext ctx)
    {
        if(tested_u > uSampleTarget) {
            currentSample += 1;
            WriteSample(ctx.photon, uSampleTarget, ctx.lod);
            uSampleTarget = (currentSample + rand.Next()) * g_integration_interval;
        }
        transmissibility *= tested_transmissibility.x;

        // If the density of the substrate is too high, consider this the new endpoint.
        //THIS CONDITION WILL BREAK REFLECTION WHEN DENSITY APPROACHES 1
        return ctx.uHitNext < ctx.uEscape && transmissibility > 1e-9;
    }
    
    bool EndTraversal(inout IntegrationContext ctx)
    {
        ctx.uHitCurrent = tested_u + log2(transmitPotential / transmissibility) / (log2(tested_transmissibility.x) - 1e-5);
        return true;//probability > 1e-7;
    }

    bool Bounce(inout IntegrationContext ctx, float3 albedo)
    {
        ctx.photon.Energy *= albedo * quantumScale;

        //float3 important_direction = ScatterImportanceLobed(ctx.photon.Origin);
        float4 important_direction = ScatterMaterially(ctx.photon.Origin, ctx.photon.Direction);
        ctx.photon.Direction = important_direction.xy;
        ctx.photon.Energy *= important_direction.z;

        return true;
    }
};

#endif