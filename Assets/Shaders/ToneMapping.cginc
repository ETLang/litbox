struct ToneMappingShape_UE5 {
    float exposure;
    float3 white_point;
    float3 black_point;
};

ToneMappingShape_UE5 ToneMap_UE5_DefaultShape() {
    ToneMappingShape_UE5 ret = { 0, (2).xxx, (-4).xxx };
    return ret;
}

// Analogous to UE5's standard tone mapping.
// Good general-purpose curve, but it makes things kinda feel like UE5...
float3 ToneMap_UE5(float3 x, ToneMappingShape_UE5 shape) {
    return smoothstep(shape.black_point, shape.white_point, log10(x) + shape.exposure);
}

float3 ToneMap_UE5(float3 x) {
    return ToneMap_UE5(x, ToneMap_UE5_DefaultShape());
}

struct ToneMappingShape_Uchimura {
    float contrast;
    float linear_base;
    float linear_span;
    float3 black_tightness;
    float3 black_pedestal;
    float maximum_brightness;
};

ToneMappingShape_Uchimura ToneMap_Uchimura_DefaultShape() {
    ToneMappingShape_Uchimura ret = { 1, 0.22f, 0.4f, (1.33).xxx, (0).xxx, 1 };
    return ret;
}

// GT Tonemapping by Hajime Uchimura (simplified)
// Good for working with 2D sprites, as it retains a linear curve where you want it.
float3 ToneMap_Uchimura(float3 x, ToneMappingShape_Uchimura shape) {
    float a = shape.contrast;
    float m = shape.linear_base;
    float l = shape.linear_span;
    float3 c = shape.black_tightness;
    float3 b = shape.black_pedestal;
    float P = shape.maximum_brightness;

    float l0 = (P - m) * l / a;
    float L0 = m - m/a;
    float L1 = m + (1-m)/a;
    float S0 = m + l0;
    float S1 = m + a * l0;  
    float C2 = (a * P) / (P - S1);
    float CP = -C2 / P;

    float3 w0 = 1.0 - smoothstep(0.0, m, x); // Toe weight
    float3 w2 = step(m + l0, x);             // Shoulder weight
    float3 w1 = 1.0 - w0 - w2;               // Linear weight

    float3 T = m * pow(x / m, c) + b;
    float3 L = m + a * (x - m);
    float3 S = P - (P - S1) * exp(CP * (x - S0));

    //return S;
    return T * w0 + L * w1 + S * w2;
}

float3 ToneMap_Uchimura(float3 x) {
    return ToneMap_Uchimura(x, ToneMap_Uchimura_DefaultShape());
}