static void avc_luma_vt_4w_msa(const uint8_t *src, int32_t src_stride,

                               uint8_t *dst, int32_t dst_stride,

                               int32_t height)

{

    int32_t loop_cnt;

    int16_t filt_const0 = 0xfb01;

    int16_t filt_const1 = 0x1414;

    int16_t filt_const2 = 0x1fb;

    v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8;

    v16i8 src10_r, src32_r, src54_r, src76_r, src21_r, src43_r, src65_r;

    v16i8 src87_r, src2110, src4332, src6554, src8776;

    v16i8 filt0, filt1, filt2;

    v8i16 out10, out32;

    v16u8 out;



    filt0 = (v16i8) __msa_fill_h(filt_const0);

    filt1 = (v16i8) __msa_fill_h(filt_const1);

    filt2 = (v16i8) __msa_fill_h(filt_const2);



    LD_SB5(src, src_stride, src0, src1, src2, src3, src4);

    src += (5 * src_stride);



    ILVR_B4_SB(src1, src0, src2, src1, src3, src2, src4, src3,

               src10_r, src21_r, src32_r, src43_r);

    ILVR_D2_SB(src21_r, src10_r, src43_r, src32_r, src2110, src4332);

    XORI_B2_128_SB(src2110, src4332);



    for (loop_cnt = (height >> 2); loop_cnt--;) {

        LD_SB4(src, src_stride, src5, src6, src7, src8);

        src += (4 * src_stride);



        ILVR_B4_SB(src5, src4, src6, src5, src7, src6, src8, src7,

                   src54_r, src65_r, src76_r, src87_r);

        ILVR_D2_SB(src65_r, src54_r, src87_r, src76_r, src6554, src8776);

        XORI_B2_128_SB(src6554, src8776);

        out10 = DPADD_SH3_SH(src2110, src4332, src6554, filt0, filt1, filt2);

        out32 = DPADD_SH3_SH(src4332, src6554, src8776, filt0, filt1, filt2);

        SRARI_H2_SH(out10, out32, 5);

        SAT_SH2_SH(out10, out32, 7);

        out = PCKEV_XORI128_UB(out10, out32);

        ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);



        dst += (4 * dst_stride);

        src2110 = src6554;

        src4332 = src8776;

        src4 = src8;

    }

}
