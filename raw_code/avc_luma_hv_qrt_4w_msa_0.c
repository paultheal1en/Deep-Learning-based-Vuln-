static void avc_luma_hv_qrt_4w_msa(const uint8_t *src_x, const uint8_t *src_y,

                                   int32_t src_stride, uint8_t *dst,

                                   int32_t dst_stride, int32_t height)

{

    uint32_t loop_cnt;

    v16i8 src_hz0, src_hz1, src_hz2, src_hz3;

    v16i8 src_vt0, src_vt1, src_vt2, src_vt3, src_vt4;

    v16i8 src_vt5, src_vt6, src_vt7, src_vt8;

    v16i8 mask0, mask1, mask2;

    v8i16 hz_out0, hz_out1, vert_out0, vert_out1;

    v8i16 out0, out1;

    v16u8 out;



    LD_SB3(&luma_mask_arr[48], 16, mask0, mask1, mask2);



    LD_SB5(src_y, src_stride, src_vt0, src_vt1, src_vt2, src_vt3, src_vt4);

    src_y += (5 * src_stride);



    src_vt0 = (v16i8) __msa_insve_w((v4i32) src_vt0, 1, (v4i32) src_vt1);

    src_vt1 = (v16i8) __msa_insve_w((v4i32) src_vt1, 1, (v4i32) src_vt2);

    src_vt2 = (v16i8) __msa_insve_w((v4i32) src_vt2, 1, (v4i32) src_vt3);

    src_vt3 = (v16i8) __msa_insve_w((v4i32) src_vt3, 1, (v4i32) src_vt4);



    XORI_B4_128_SB(src_vt0, src_vt1, src_vt2, src_vt3);



    for (loop_cnt = (height >> 2); loop_cnt--;) {

        LD_SB4(src_x, src_stride, src_hz0, src_hz1, src_hz2, src_hz3);

        src_x += (4 * src_stride);



        XORI_B4_128_SB(src_hz0, src_hz1, src_hz2, src_hz3);



        hz_out0 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src_hz0,

                                                              src_hz1, mask0,

                                                              mask1, mask2);

        hz_out1 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src_hz2,

                                                              src_hz3, mask0,

                                                              mask1, mask2);



        SRARI_H2_SH(hz_out0, hz_out1, 5);

        SAT_SH2_SH(hz_out0, hz_out1, 7);



        LD_SB4(src_y, src_stride, src_vt5, src_vt6, src_vt7, src_vt8);

        src_y += (4 * src_stride);



        src_vt4 = (v16i8) __msa_insve_w((v4i32) src_vt4, 1, (v4i32) src_vt5);

        src_vt5 = (v16i8) __msa_insve_w((v4i32) src_vt5, 1, (v4i32) src_vt6);

        src_vt6 = (v16i8) __msa_insve_w((v4i32) src_vt6, 1, (v4i32) src_vt7);

        src_vt7 = (v16i8) __msa_insve_w((v4i32) src_vt7, 1, (v4i32) src_vt8);



        XORI_B4_128_SB(src_vt4, src_vt5, src_vt6, src_vt7);



        /* filter calc */

        vert_out0 = AVC_CALC_DPADD_B_6PIX_2COEFF_R_SH(src_vt0, src_vt1,

                                                      src_vt2, src_vt3,

                                                      src_vt4, src_vt5);

        vert_out1 = AVC_CALC_DPADD_B_6PIX_2COEFF_R_SH(src_vt2, src_vt3,

                                                      src_vt4, src_vt5,

                                                      src_vt6, src_vt7);



        SRARI_H2_SH(vert_out0, vert_out1, 5);

        SAT_SH2_SH(vert_out0, vert_out1, 7);



        out0 = __msa_srari_h((hz_out0 + vert_out0), 1);

        out1 = __msa_srari_h((hz_out1 + vert_out1), 1);



        SAT_SH2_SH(out0, out1, 7);

        out = PCKEV_XORI128_UB(out0, out1);

        ST4x4_UB(out, out, 0, 1, 2, 3, dst, dst_stride);

        dst += (4 * dst_stride);



        src_vt3 = src_vt7;

        src_vt1 = src_vt5;

        src_vt0 = src_vt4;

        src_vt4 = src_vt8;

        src_vt2 = src_vt6;

    }

}
