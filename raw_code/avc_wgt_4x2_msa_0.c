static void avc_wgt_4x2_msa(uint8_t *data,

                            int32_t stride,

                            int32_t log2_denom,

                            int32_t src_weight,

                            int32_t offset_in)

{

    uint32_t data0, data1;

    v16u8 zero = { 0 };

    v16u8 src0, src1;

    v4i32 res0, res1;

    v8i16 temp0, temp1;

    v16u8 vec0, vec1;

    v8i16 wgt, denom, offset;



    offset_in <<= (log2_denom);



    if (log2_denom) {

        offset_in += (1 << (log2_denom - 1));

    }



    wgt = __msa_fill_h(src_weight);

    offset = __msa_fill_h(offset_in);

    denom = __msa_fill_h(log2_denom);



    data0 = LOAD_WORD(data);

    data1 = LOAD_WORD(data + stride);



    src0 = (v16u8) __msa_fill_w(data0);

    src1 = (v16u8) __msa_fill_w(data1);



    ILVR_B_2VECS_UB(src0, src1, zero, zero, vec0, vec1);



    temp0 = wgt * (v8i16) vec0;

    temp1 = wgt * (v8i16) vec1;



    temp0 = __msa_adds_s_h(temp0, offset);

    temp1 = __msa_adds_s_h(temp1, offset);



    temp0 = __msa_maxi_s_h(temp0, 0);

    temp1 = __msa_maxi_s_h(temp1, 0);



    temp0 = __msa_srl_h(temp0, denom);

    temp1 = __msa_srl_h(temp1, denom);



    temp0 = (v8i16) __msa_sat_u_h((v8u16) temp0, 7);

    temp1 = (v8i16) __msa_sat_u_h((v8u16) temp1, 7);



    res0 = (v4i32) __msa_pckev_b((v16i8) temp0, (v16i8) temp0);

    res1 = (v4i32) __msa_pckev_b((v16i8) temp1, (v16i8) temp1);



    data0 = __msa_copy_u_w(res0, 0);

    data1 = __msa_copy_u_w(res1, 0);



    STORE_WORD(data, data0);

    data += stride;

    STORE_WORD(data, data1);

}
