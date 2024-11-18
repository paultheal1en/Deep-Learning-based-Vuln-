static int dnxhd_10bit_dct_quantize(MpegEncContext *ctx, DCTELEM *block,

                                    int n, int qscale, int *overflow)

{

    const uint8_t *scantable= ctx->intra_scantable.scantable;

    const int *qmat = ctx->q_intra_matrix[qscale];

    int last_non_zero = 0;



    ctx->dsp.fdct(block);



    // Divide by 4 with rounding, to compensate scaling of DCT coefficients

    block[0] = (block[0] + 2) >> 2;



    for (int i = 1; i < 64; ++i) {

        int j = scantable[i];

        int sign = block[j] >> 31;

        int level = (block[j] ^ sign) - sign;

        level = level * qmat[j] >> DNX10BIT_QMAT_SHIFT;

        block[j] = (level ^ sign) - sign;

        if (level)

            last_non_zero = i;

    }



    return last_non_zero;

}
