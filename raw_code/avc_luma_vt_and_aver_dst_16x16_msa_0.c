static void avc_luma_vt_and_aver_dst_16x16_msa(const uint8_t *src,

                                               int32_t src_stride,

                                               uint8_t *dst, int32_t dst_stride)

{

    int32_t loop_cnt;

    int16_t filt_const0 = 0xfb01;

    int16_t filt_const1 = 0x1414;

    int16_t filt_const2 = 0x1fb;

    v16u8 dst0, dst1, dst2, dst3;

    v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8;

    v16i8 src10_r, src32_r, src54_r, src76_r, src21_r, src43_r, src65_r;

    v16i8 src87_r, src10_l, src32_l, src54_l, src76_l, src21_l, src43_l;

    v16i8 src65_l, src87_l;

    v8i16 out0_r, out1_r, out2_r, out3_r, out0_l, out1_l, out2_l, out3_l;

    v16i8 filt0, filt1, filt2;

    v16u8 res0, res1, res2, res3;



    filt0 = (v16i8) __msa_fill_h(filt_const0);

    filt1 = (v16i8) __msa_fill_h(filt_const1);

    filt2 = (v16i8) __msa_fill_h(filt_const2);



    LD_SB5(src, src_stride, src0, src1, src2, src3, src4);

    src += (5 * src_stride);



    XORI_B5_128_SB(src0, src1, src2, src3, src4);

    ILVR_B4_SB(src1, src0, src2, src1, src3, src2, src4, src3,

               src10_r, src21_r, src32_r, src43_r);

    ILVL_B4_SB(src1, src0, src2, src1, src3, src2, src4, src3,

               src10_l, src21_l, src32_l, src43_l);



    for (loop_cnt = 4; loop_cnt--;) {

        LD_SB4(src, src_stride, src5, src6, src7, src8);

        src += (4 * src_stride);



        XORI_B4_128_SB(src5, src6, src7, src8);

        ILVR_B4_SB(src5, src4, src6, src5, src7, src6, src8, src7,

                   src54_r, src65_r, src76_r, src87_r);

        ILVL_B4_SB(src5, src4, src6, src5, src7, src6, src8, src7,

                   src54_l, src65_l, src76_l, src87_l);

        out0_r = DPADD_SH3_SH(src10_r, src32_r, src54_r, filt0, filt1, filt2);

        out1_r = DPADD_SH3_SH(src21_r, src43_r, src65_r, filt0, filt1, filt2);

        out2_r = DPADD_SH3_SH(src32_r, src54_r, src76_r, filt0, filt1, filt2);

        out3_r = DPADD_SH3_SH(src43_r, src65_r, src87_r, filt0, filt1, filt2);

        out0_l = DPADD_SH3_SH(src10_l, src32_l, src54_l, filt0, filt1, filt2);

        out1_l = DPADD_SH3_SH(src21_l, src43_l, src65_l, filt0, filt1, filt2);

        out2_l = DPADD_SH3_SH(src32_l, src54_l, src76_l, filt0, filt1, filt2);

        out3_l = DPADD_SH3_SH(src43_l, src65_l, src87_l, filt0, filt1, filt2);

        SRARI_H4_SH(out0_r, out1_r, out2_r, out3_r, 5);

        SRARI_H4_SH(out0_l, out1_l, out2_l, out3_l, 5);

        SAT_SH4_SH(out0_r, out1_r, out2_r, out3_r, 7);

        SAT_SH4_SH(out0_l, out1_l, out2_l, out3_l, 7);

        LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);

        PCKEV_B4_UB(out0_l, out0_r, out1_l, out1_r, out2_l, out2_r, out3_l,

                    out3_r, res0, res1, res2, res3);

        XORI_B4_128_UB(res0, res1, res2, res3);

        AVER_UB4_UB(res0, dst0, res1, dst1, res2, dst2, res3, dst3,

                    res0, res1, res2, res3);

        ST_UB4(res0, res1, res2, res3, dst, dst_stride);

        dst += (4 * dst_stride);



        src10_r = src54_r;

        src32_r = src76_r;

        src21_r = src65_r;

        src43_r = src87_r;

        src10_l = src54_l;

        src32_l = src76_l;

        src21_l = src65_l;

        src43_l = src87_l;

        src4 = src8;

    }

}
