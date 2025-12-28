struct ToneMappingShape {
    float exposure;
    float3 white_point;
    float3 black_point;
};

// Analogous to UE5's standard tone mapping.
// Good general-purpose curve, but it makes things kinda feel like UE5...
float3 ToneMap_UE5(float3 x, ToneMappingShape shape) {
    return smoothstep(shape.black_point, shape.white_point, log10(x) + shape.exposure);
}

float3 ToneMap_UE5(float3 x) {
    ToneMappingShape default_shape = { 0, (2).xxx, (-4).xxx };
    return ToneMap_UE5(x, default_shape);
}


