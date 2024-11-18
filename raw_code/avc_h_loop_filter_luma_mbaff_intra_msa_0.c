static void avc_h_loop_filter_luma_mbaff_intra_msa(uint8_t *src,

                                                   int32_t stride,

                                                   int32_t alpha_in,

                                                   int32_t beta_in)

{

    uint64_t load0, load1;

    uint32_t out0, out2;

    uint16_t out1, out3;

    v8u16 src0_r, src1_r, src2_r, src3_r, src4_r, src5_r, src6_r, src7_r;

    v8u16 dst0_r, dst1_r, dst4_r, dst5_r;

    v8u16 dst2_x_r, dst2_y_r, dst3_x_r, dst3_y_r;

    v16u8 dst0, dst1, dst4, dst5, dst2_x, dst2_y, dst3_x, dst3_y;

    v8i16 tmp0, tmp1, tmp2, tmp3;

    v16u8 alpha, beta;

    v16u8 p0_asub_q0, p1_asub_p0, q1_asub_q0, p2_asub_p0, q2_asub_q0;

    v16u8 is_less_than, is_less_than_alpha, is_less_than_beta;

    v16u8 is_less_than_beta1, is_less_than_beta2;

    v16i8 src0 = { 0 };

    v16i8 src1 = { 0 };

    v16i8 src2 = { 0 };

    v16i8 src3 = { 0 };

    v16i8 src4 = { 0 };

    v16i8 src5 = { 0 };

    v16i8 src6 = { 0 };

    v16i8 src7 = { 0 };

    v16i8 zeros = { 0 };



    load0 = LOAD_DWORD(src - 4);

    load1 = LOAD_DWORD(src + stride - 4);

    src0 = (v16i8) __msa_insert_d((v2i64) src0, 0, load0);

    src1 = (v16i8) __msa_insert_d((v2i64) src1, 0, load1);



    load0 = LOAD_DWORD(src + (2 * stride) - 4);

    load1 = LOAD_DWORD(src + (3 * stride) - 4);

    src2 = (v16i8) __msa_insert_d((v2i64) src2, 0, load0);

    src3 = (v16i8) __msa_insert_d((v2i64) src3, 0, load1);



    load0 = LOAD_DWORD(src + (4 * stride) - 4);

    load1 = LOAD_DWORD(src + (5 * stride) - 4);

    src4 = (v16i8) __msa_insert_d((v2i64) src4, 0, load0);

    src5 = (v16i8) __msa_insert_d((v2i64) src5, 0, load1);



    load0 = LOAD_DWORD(src + (6 * stride) - 4);

    load1 = LOAD_DWORD(src + (7 * stride) - 4);

    src6 = (v16i8) __msa_insert_d((v2i64) src6, 0, load0);

    src7 = (v16i8) __msa_insert_d((v2i64) src7, 0, load1);



    src0 = __msa_ilvr_b(src1, src0);

    src1 = __msa_ilvr_b(src3, src2);

    src2 = __msa_ilvr_b(src5, src4);

    src3 = __msa_ilvr_b(src7, src6);

    tmp0 = __msa_ilvr_h((v8i16) src1, (v8i16) src0);

    tmp1 = __msa_ilvl_h((v8i16) src1, (v8i16) src0);

    tmp2 = __msa_ilvr_h((v8i16) src3, (v8i16) src2);

    tmp3 = __msa_ilvl_h((v8i16) src3, (v8i16) src2);

    src6 = (v16i8) __msa_ilvr_w((v4i32) tmp2, (v4i32) tmp0);

    src0 = __msa_sldi_b(zeros, src6, 8);

    src1 = (v16i8) __msa_ilvl_w((v4i32) tmp2, (v4i32) tmp0);

    src2 = __msa_sldi_b(zeros, src1, 8);

    src3 = (v16i8) __msa_ilvr_w((v4i32) tmp3, (v4i32) tmp1);

    src4 = __msa_sldi_b(zeros, src3, 8);

    src5 = (v16i8) __msa_ilvl_w((v4i32) tmp3, (v4i32) tmp1);

    src7 = __msa_sldi_b(zeros, src5, 8);



    p0_asub_q0 = __msa_asub_u_b((v16u8) src2, (v16u8) src3);

    p1_asub_p0 = __msa_asub_u_b((v16u8) src1, (v16u8) src2);

    q1_asub_q0 = __msa_asub_u_b((v16u8) src4, (v16u8) src3);



    alpha = (v16u8) __msa_fill_b(alpha_in);

    beta = (v16u8) __msa_fill_b(beta_in);



    is_less_than_alpha = (p0_asub_q0 < alpha);

    is_less_than_beta = (p1_asub_p0 < beta);

    is_less_than = is_less_than_alpha & is_less_than_beta;

    is_less_than_beta = (q1_asub_q0 < beta);

    is_less_than = is_less_than & is_less_than_beta;



    alpha >>= 2;

    alpha += 2;



    is_less_than_alpha = (p0_asub_q0 < alpha);



    p2_asub_p0 = __msa_asub_u_b((v16u8) src0, (v16u8) src2);

    is_less_than_beta1 = (p2_asub_p0 < beta);

    q2_asub_q0 = __msa_asub_u_b((v16u8) src5, (v16u8) src3);

    is_less_than_beta2 = (q2_asub_q0 < beta);



    src0_r = (v8u16) __msa_ilvr_b(zeros, src0);

    src1_r = (v8u16) __msa_ilvr_b(zeros, src1);

    src2_r = (v8u16) __msa_ilvr_b(zeros, src2);

    src3_r = (v8u16) __msa_ilvr_b(zeros, src3);

    src4_r = (v8u16) __msa_ilvr_b(zeros, src4);

    src5_r = (v8u16) __msa_ilvr_b(zeros, src5);

    src6_r = (v8u16) __msa_ilvr_b(zeros, src6);

    src7_r = (v8u16) __msa_ilvr_b(zeros, src7);



    dst2_x_r = src1_r + src2_r + src3_r;

    dst2_x_r = src0_r + (2 * (dst2_x_r)) + src4_r;

    dst2_x_r = (v8u16) __msa_srari_h((v8i16) dst2_x_r, 3);



    dst1_r = src0_r + src1_r + src2_r + src3_r;

    dst1_r = (v8u16) __msa_srari_h((v8i16) dst1_r, 2);



    dst0_r = (2 * src6_r) + (3 * src0_r);

    dst0_r += src1_r + src2_r + src3_r;

    dst0_r = (v8u16) __msa_srari_h((v8i16) dst0_r, 3);



    dst2_y_r = (2 * src1_r) + src2_r + src4_r;

    dst2_y_r = (v8u16) __msa_srari_h((v8i16) dst2_y_r, 2);



    dst2_x = (v16u8) __msa_pckev_b((v16i8) dst2_x_r, (v16i8) dst2_x_r);

    dst2_y = (v16u8) __msa_pckev_b((v16i8) dst2_y_r, (v16i8) dst2_y_r);

    dst2_x = __msa_bmnz_v(dst2_y, dst2_x, is_less_than_beta1);



    dst3_x_r = src2_r + src3_r + src4_r;

    dst3_x_r = src1_r + (2 * dst3_x_r) + src5_r;

    dst3_x_r = (v8u16) __msa_srari_h((v8i16) dst3_x_r, 3);



    dst4_r = src2_r + src3_r + src4_r + src5_r;

    dst4_r = (v8u16) __msa_srari_h((v8i16) dst4_r, 2);



    dst5_r = (2 * src7_r) + (3 * src5_r);

    dst5_r += src4_r + src3_r + src2_r;

    dst5_r = (v8u16) __msa_srari_h((v8i16) dst5_r, 3);



    dst3_y_r = (2 * src4_r) + src3_r + src1_r;

    dst3_y_r = (v8u16) __msa_srari_h((v8i16) dst3_y_r, 2);

    dst3_x = (v16u8) __msa_pckev_b((v16i8) dst3_x_r, (v16i8) dst3_x_r);

    dst3_y = (v16u8) __msa_pckev_b((v16i8) dst3_y_r, (v16i8) dst3_y_r);

    dst3_x = __msa_bmnz_v(dst3_y, dst3_x, is_less_than_beta2);



    dst2_y_r = (2 * src1_r) + src2_r + src4_r;

    dst2_y_r = (v8u16) __msa_srari_h((v8i16) dst2_y_r, 2);



    dst3_y_r = (2 * src4_r) + src3_r + src1_r;

    dst3_y_r = (v8u16) __msa_srari_h((v8i16) dst3_y_r, 2);



    dst2_y = (v16u8) __msa_pckev_b((v16i8) dst2_y_r, (v16i8) dst2_y_r);

    dst3_y = (v16u8) __msa_pckev_b((v16i8) dst3_y_r, (v16i8) dst3_y_r);

    dst2_x = __msa_bmnz_v(dst2_y, dst2_x, is_less_than_alpha);

    dst3_x = __msa_bmnz_v(dst3_y, dst3_x, is_less_than_alpha);

    dst2_x = __msa_bmnz_v((v16u8) src2, dst2_x, is_less_than);

    dst3_x = __msa_bmnz_v((v16u8) src3, dst3_x, is_less_than);



    is_less_than = is_less_than_alpha & is_less_than;

    dst1 = (v16u8) __msa_pckev_b((v16i8) dst1_r, (v16i8) dst1_r);

    is_less_than_beta1 = is_less_than_beta1 & is_less_than;

    dst1 = __msa_bmnz_v((v16u8) src1, dst1, is_less_than_beta1);



    dst0 = (v16u8) __msa_pckev_b((v16i8) dst0_r, (v16i8) dst0_r);

    dst0 = __msa_bmnz_v((v16u8) src0, dst0, is_less_than_beta1);



    dst4 = (v16u8) __msa_pckev_b((v16i8) dst4_r, (v16i8) dst4_r);

    is_less_than_beta2 = is_less_than_beta2 & is_less_than;

    dst4 = __msa_bmnz_v((v16u8) src4, dst4, is_less_than_beta2);



    dst5 = (v16u8) __msa_pckev_b((v16i8) dst5_r, (v16i8) dst5_r);

    dst5 = __msa_bmnz_v((v16u8) src5, dst5, is_less_than_beta2);



    dst0 = (v16u8) __msa_ilvr_b((v16i8) dst1, (v16i8) dst0);

    dst1 = (v16u8) __msa_ilvr_b((v16i8) dst3_x, (v16i8) dst2_x);

    dst2_x = (v16u8) __msa_ilvr_b((v16i8) dst5, (v16i8) dst4);

    tmp0 = __msa_ilvr_h((v8i16) dst1, (v8i16) dst0);

    tmp1 = __msa_ilvl_h((v8i16) dst1, (v8i16) dst0);

    tmp2 = __msa_ilvr_h((v8i16) zeros, (v8i16) dst2_x);

    tmp3 = __msa_ilvl_h((v8i16) zeros, (v8i16) dst2_x);

    dst0 = (v16u8) __msa_ilvr_w((v4i32) tmp2, (v4i32) tmp0);

    dst1 = (v16u8) __msa_sldi_b(zeros, (v16i8) dst0, 8);

    dst2_x = (v16u8) __msa_ilvl_w((v4i32) tmp2, (v4i32) tmp0);

    dst3_x = (v16u8) __msa_sldi_b(zeros, (v16i8) dst2_x, 8);

    dst4 = (v16u8) __msa_ilvr_w((v4i32) tmp3, (v4i32) tmp1);

    dst5 = (v16u8) __msa_sldi_b(zeros, (v16i8) dst4, 8);

    dst2_y = (v16u8) __msa_ilvl_w((v4i32) tmp3, (v4i32) tmp1);

    dst3_y = (v16u8) __msa_sldi_b(zeros, (v16i8) dst2_y, 8);



    out0 = __msa_copy_u_w((v4i32) dst0, 0);

    out1 = __msa_copy_u_h((v8i16) dst0, 2);

    out2 = __msa_copy_u_w((v4i32) dst1, 0);

    out3 = __msa_copy_u_h((v8i16) dst1, 2);



    STORE_WORD((src - 3), out0);

    STORE_HWORD((src + 1), out1);

    src += stride;

    STORE_WORD((src - 3), out2);

    STORE_HWORD((src + 1), out3);

    src += stride;



    out0 = __msa_copy_u_w((v4i32) dst2_x, 0);

    out1 = __msa_copy_u_h((v8i16) dst2_x, 2);

    out2 = __msa_copy_u_w((v4i32) dst3_x, 0);

    out3 = __msa_copy_u_h((v8i16) dst3_x, 2);



    STORE_WORD((src - 3), out0);

    STORE_HWORD((src + 1), out1);

    src += stride;

    STORE_WORD((src - 3), out2);

    STORE_HWORD((src + 1), out3);

    src += stride;



    out0 = __msa_copy_u_w((v4i32) dst4, 0);

    out1 = __msa_copy_u_h((v8i16) dst4, 2);

    out2 = __msa_copy_u_w((v4i32) dst5, 0);

    out3 = __msa_copy_u_h((v8i16) dst5, 2);



    STORE_WORD((src - 3), out0);

    STORE_HWORD((src + 1), out1);

    src += stride;

    STORE_WORD((src - 3), out2);

    STORE_HWORD((src + 1), out3);

    src += stride;



    out0 = __msa_copy_u_w((v4i32) dst2_y, 0);

    out1 = __msa_copy_u_h((v8i16) dst2_y, 2);

    out2 = __msa_copy_u_w((v4i32) dst3_y, 0);

    out3 = __msa_copy_u_h((v8i16) dst3_y, 2);



    STORE_WORD((src - 3), out0);

    STORE_HWORD((src + 1), out1);

    src += stride;

    STORE_WORD((src - 3), out2);

    STORE_HWORD((src + 1), out3);

}
