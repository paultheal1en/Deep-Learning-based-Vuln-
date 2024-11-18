static void intra_predict_mad_cow_dc_0l0_8x8_msa(uint8_t *src, int32_t stride)

{

    uint8_t lp_cnt;

    uint32_t src0 = 0;

    uint64_t out0, out1;



    for (lp_cnt = 0; lp_cnt < 4; lp_cnt++) {

        src0 += src[(4 + lp_cnt) * stride - 1];

    }



    src0 = (src0 + 2) >> 2;



    out0 = 0x8080808080808080;

    out1 = src0 * 0x0101010101010101;



    for (lp_cnt = 4; lp_cnt--;) {

        SD(out0, src);

        SD(out1, src + stride * 4);

        src += stride;

    }

}
