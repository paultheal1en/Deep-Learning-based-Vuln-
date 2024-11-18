static void avc_luma_mid_and_aver_dst_4x4_msa(const uint8_t *src,

                                              int32_t src_stride,

                                              uint8_t *dst, int32_t dst_stride)

{

    v16i8 src0, src1, src2, src3, src4;

    v16i8 mask0, mask1, mask2;

    v8i16 hz_out0, hz_out1, hz_out2, hz_out3;

    v8i16 hz_out4, hz_out5, hz_out6, hz_out7, hz_out8;

    v8i16 res0, res1, res2, res3;

    v16u8 dst0, dst1, dst2, dst3;

    v16u8 tmp0, tmp1, tmp2, tmp3;



    LD_SB3(&luma_mask_arr[48], 16, mask0, mask1, mask2);

    LD_SB5(src, src_stride, src0, src1, src2, src3, src4);

    src += (5 * src_stride);



    XORI_B5_128_SB(src0, src1, src2, src3, src4);



    hz_out0 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src0, src1,

                                                          mask0, mask1, mask2);

    hz_out2 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src2, src3,

                                                          mask0, mask1, mask2);



    PCKOD_D2_SH(hz_out0, hz_out0, hz_out2, hz_out2, hz_out1, hz_out3);



    hz_out4 = AVC_HORZ_FILTER_SH(src4, src4, mask0, mask1, mask2);



    LD_SB4(src, src_stride, src0, src1, src2, src3);

    XORI_B4_128_SB(src0, src1, src2, src3);



    hz_out5 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src0, src1,

                                                          mask0, mask1, mask2);

    hz_out7 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src2, src3,

                                                          mask0, mask1, mask2);



    PCKOD_D2_SH(hz_out5, hz_out5, hz_out7, hz_out7, hz_out6, hz_out8);



    res0 = AVC_CALC_DPADD_H_6PIX_2COEFF_R_SH(hz_out0, hz_out1, hz_out2,

                                             hz_out3, hz_out4, hz_out5);

    res1 = AVC_CALC_DPADD_H_6PIX_2COEFF_R_SH(hz_out1, hz_out2, hz_out3,

                                             hz_out4, hz_out5, hz_out6);

    res2 = AVC_CALC_DPADD_H_6PIX_2COEFF_R_SH(hz_out2, hz_out3, hz_out4,

                                             hz_out5, hz_out6, hz_out7);

    res3 = AVC_CALC_DPADD_H_6PIX_2COEFF_R_SH(hz_out3, hz_out4, hz_out5,

                                             hz_out6, hz_out7, hz_out8);

    LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);

    tmp0 = PCKEV_XORI128_UB(res0, res1);

    tmp1 = PCKEV_XORI128_UB(res2, res3);

    PCKEV_D2_UB(dst1, dst0, dst3, dst2, tmp2, tmp3);

    AVER_UB2_UB(tmp0, tmp2, tmp1, tmp3, tmp0, tmp1);



    ST4x4_UB(tmp0, tmp1, 0, 2, 0, 2, dst, dst_stride);

}
