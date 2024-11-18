static int fic_decode_block(FICContext *ctx, GetBitContext *gb,

                            uint8_t *dst, int stride, int16_t *block, int *is_p)

{

    int i, num_coeff;



    /* Is it a skip block? */

    if (get_bits1(gb)) {

        *is_p = 1;

        return 0;

    }



    memset(block, 0, sizeof(*block) * 64);



    num_coeff = get_bits(gb, 7);

    if (num_coeff > 64)

        return AVERROR_INVALIDDATA;



    for (i = 0; i < num_coeff; i++)

        block[ff_zigzag_direct[i]] = get_se_golomb(gb) *

                                     ctx->qmat[ff_zigzag_direct[i]];



    fic_idct_put(dst, stride, block);



    return 0;

}
