static void quantize_and_encode_band_cost_ZERO_mips(struct AACEncContext *s,

                                                         PutBitContext *pb, const float *in, float *out,

                                                         const float *scaled, int size, int scale_idx,

                                                         int cb, const float lambda, const float uplim,

                                                         int *bits, const float ROUNDING) {

    int i;

    if (bits)

        *bits = 0;

    if (out) {

        for (i = 0; i < size; i += 4) {

           out[i  ] = 0.0f;

           out[i+1] = 0.0f;

           out[i+2] = 0.0f;

           out[i+3] = 0.0f;

        }

    }

}
