static void avc_wgt_4x4multiple_msa(uint8_t *data,

                                    int32_t stride,

                                    int32_t height,

                                    int32_t log2_denom,

                                    int32_t src_weight,

                                    int32_t offset_in)

{

    uint8_t cnt;

    uint32_t data0, data1, data2, data3;

    v16u8 zero = { 0 };

    v16u8 src0, src1, src2, src3;

    v8u16 temp0, temp1, temp2, temp3;

    v8i16 wgt, denom, offset;



    offset_in <<= (log2_denom);



    if (log2_denom) {

        offset_in += (1 << (log2_denom - 1));

    }



    wgt = __msa_fill_h(src_weight);

    offset = __msa_fill_h(offset_in);

    denom = __msa_fill_h(log2_denom);



    for (cnt = height / 4; cnt--;) {

        LOAD_4WORDS_WITH_STRIDE(data, stride, data0, data1, data2, data3);



        src0 = (v16u8) __msa_fill_w(data0);

        src1 = (v16u8) __msa_fill_w(data1);

        src2 = (v16u8) __msa_fill_w(data2);

        src3 = (v16u8) __msa_fill_w(data3);



        ILVR_B_4VECS_UH(src0, src1, src2, src3, zero, zero, zero, zero,

                        temp0, temp1, temp2, temp3);



        temp0 *= wgt;

        temp1 *= wgt;

        temp2 *= wgt;

        temp3 *= wgt;



        ADDS_S_H_4VECS_UH(temp0, offset, temp1, offset,

                          temp2, offset, temp3, offset,

                          temp0, temp1, temp2, temp3);



        MAXI_S_H_4VECS_UH(temp0, temp1, temp2, temp3, 0);



        SRL_H_4VECS_UH(temp0, temp1, temp2, temp3,

                       temp0, temp1, temp2, temp3, denom);



        SAT_U_H_4VECS_UH(temp0, temp1, temp2, temp3, 7);



        PCKEV_B_STORE_4_BYTES_4(temp0, temp1, temp2, temp3, data, stride);

        data += (4 * stride);

    }

}
