uint32_t HELPER(neon_abd_f32)(uint32_t a, uint32_t b)

{

    float32 f0 = make_float32(a);

    float32 f1 = make_float32(b);

    return float32_val((float32_compare_quiet(f0, f1, NFS) == 1)

                    ? float32_sub(f0, f1, NFS)

                    : float32_sub(f1, f0, NFS));

}
