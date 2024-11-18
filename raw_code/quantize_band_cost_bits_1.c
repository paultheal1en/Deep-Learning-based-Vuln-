static float quantize_band_cost_bits(struct AACEncContext *s, const float *in,

                                     const float *scaled, int size, int scale_idx,

                                     int cb, const float lambda, const float uplim,

                                     int *bits, int rtz)

{

    return get_band_numbits(s, NULL, in, scaled, size, scale_idx, cb, lambda, uplim, bits);

}
