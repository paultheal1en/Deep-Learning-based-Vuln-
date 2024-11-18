static void avc_biwgt_4x2_msa(uint8_t *src,

                              int32_t src_stride,

                              uint8_t *dst,

                              int32_t dst_stride,

                              int32_t log2_denom,

                              int32_t src_weight,

                              int32_t dst_weight,

                              int32_t offset_in)

{

    uint32_t load0, load1, out0, out1;

    v16i8 src_wgt, dst_wgt, wgt;

    v16i8 src0, src1, dst0, dst1;

    v8i16 temp0, temp1, denom, offset, add_val;

    int32_t val = 128 * (src_weight + dst_weight);



    offset_in = ((offset_in + 1) | 1) << log2_denom;



    src_wgt = __msa_fill_b(src_weight);

    dst_wgt = __msa_fill_b(dst_weight);

    offset = __msa_fill_h(offset_in);

    denom = __msa_fill_h(log2_denom + 1);

    add_val = __msa_fill_h(val);

    offset += add_val;



    wgt = __msa_ilvev_b(dst_wgt, src_wgt);



    load0 = LOAD_WORD(src);

    src += src_stride;

    load1 = LOAD_WORD(src);



    src0 = (v16i8) __msa_fill_w(load0);

    src1 = (v16i8) __msa_fill_w(load1);



    load0 = LOAD_WORD(dst);

    load1 = LOAD_WORD(dst + dst_stride);



    dst0 = (v16i8) __msa_fill_w(load0);

    dst1 = (v16i8) __msa_fill_w(load1);



    XORI_B_4VECS_SB(src0, src1, dst0, dst1, src0, src1, dst0, dst1, 128);



    ILVR_B_2VECS_SH(src0, src1, dst0, dst1, temp0, temp1);



    temp0 = __msa_dpadd_s_h(offset, wgt, (v16i8) temp0);

    temp1 = __msa_dpadd_s_h(offset, wgt, (v16i8) temp1);



    temp0 >>= denom;

    temp1 >>= denom;



    temp0 = CLIP_UNSIGNED_CHAR_H(temp0);

    temp1 = CLIP_UNSIGNED_CHAR_H(temp1);



    dst0 = __msa_pckev_b((v16i8) temp0, (v16i8) temp0);

    dst1 = __msa_pckev_b((v16i8) temp1, (v16i8) temp1);



    out0 = __msa_copy_u_w((v4i32) dst0, 0);

    out1 = __msa_copy_u_w((v4i32) dst1, 0);



    STORE_WORD(dst, out0);

    dst += dst_stride;

    STORE_WORD(dst, out1);

}
