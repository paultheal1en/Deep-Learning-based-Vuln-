static void intra_predict_horiz_16x16_msa(uint8_t *src, int32_t src_stride,

                                          uint8_t *dst, int32_t dst_stride)

{

    uint32_t row;

    uint8_t inp0, inp1, inp2, inp3;

    v16u8 src0, src1, src2, src3;



    for (row = 4; row--;) {

        inp0 = src[0];

        src += src_stride;

        inp1 = src[0];

        src += src_stride;

        inp2 = src[0];

        src += src_stride;

        inp3 = src[0];

        src += src_stride;



        src0 = (v16u8) __msa_fill_b(inp0);

        src1 = (v16u8) __msa_fill_b(inp1);

        src2 = (v16u8) __msa_fill_b(inp2);

        src3 = (v16u8) __msa_fill_b(inp3);



        ST_UB4(src0, src1, src2, src3, dst, dst_stride);

        dst += (4 * dst_stride);

    }

}
