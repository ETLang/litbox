    void Integrate(inout Ray photon, MONTE_CARLO_MODEL state) { 
        IntegrationContext ctx;
        ctx.Init(photon);

        for(uint bounce = 0;bounce < g_bounces;) {
            float uScatter;

            if(ctx.photon.Direction.x == 0) ctx.photon.Direction.x = 1e-8;
            if(ctx.photon.Direction.y == 0) ctx.photon.Direction.y = 1e-8;
        
            float2 uvOrigin = ctx.photon.Origin / g_target_size;
            float2 uvDirection = ctx.photon.Direction / g_target_size;
            float4 uBoundaryBox = (float4(0,1,0,1) - uvOrigin.xxyy) / uvDirection.xxyy;
            ctx.uEscape = min(max(uBoundaryBox[0], uBoundaryBox[1]), max(uBoundaryBox[2], uBoundaryBox[3]));

            state.BeginTraversal(ctx);

            ctx.uHitCurrent = 0;
            bool continueRunning = true;
            for(int steps = 0;steps < 3000;steps++) {
                ctx.testUV = uvOrigin + uvDirection * ctx.uHitCurrent;
                bool overshoot = false;
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
    }
