    void Integrate(inout Ray photon, MONTE_CARLO_MODEL state) { 
        for(uint bounce = 0;bounce < g_bounces;bounce++) {
            float uScatter;

            if(photon.Direction.x == 0) photon.Direction.x = 1e-8;
            if(photon.Direction.y == 0) photon.Direction.y = 1e-8;
        
            float2 uvOrigin = photon.Origin / g_target_size;
            float2 uvDirection = photon.Direction / g_target_size;
            float4 uBoundaryBox = (float4(0,1,0,1) - uvOrigin.xxyy) / uvDirection.xxyy;
            float uEscape = min(max(uBoundaryBox[0], uBoundaryBox[1]), max(uBoundaryBox[2], uBoundaryBox[3]));

            state.BeginTraversal(uEscape, photon);

            float uHit = 0;
            float2 pSample;
            bool continueRunning = true;
            for(int steps = 0;steps < 3000;steps++) {
                float4 T;
                float uHitNext;

                pSample = uvOrigin + uvDirection * uHit;
                bool overshoot = false;
                int lod = g_quadTreeLeaves.SampleLevel(sampler_point_clamp, pSample, 0).x;

                do {
                    T = g_transmissibility.SampleLevel(samplerg_transmissibility, pSample, lod);
                    uHitNext = uHit + (1 << lod);
                    overshoot = state.Test(uHitNext, T);
                    
                    if (!overshoot) { break; }
                    if (lod <= 0) { break; }
                    lod--;
                } while (true);
            
                if(!overshoot) { // Keep propagating
                    uHit = uHitNext;
                    if(!state.Propagate()) {
                        continueRunning = false;
                        break;
                    }
                } else { // Scatter occurs within this sample, end traversal
                    continueRunning = state.EndTraversal(photon, uHit);
                    break;
                }
            }

            // if(failure) {
            //     // Fail condition... traversal took too many steps
            //     WritePhoton(photon.Origin, uint3(1000000000, 1000000000, 0));
            //     WritePhoton(pSample * g_target_size, uint3(0, 1000000000, 0));
            //     return false;,
            // }

            if(!continueRunning) break;

            photon.Origin += photon.Direction * uHit;
            float3 albedo = g_albedo.SampleLevel(sampler_point_clamp, photon.Origin / g_target_size, 0).rgb;
            state.Bounce(photon, albedo);
        }
    }
