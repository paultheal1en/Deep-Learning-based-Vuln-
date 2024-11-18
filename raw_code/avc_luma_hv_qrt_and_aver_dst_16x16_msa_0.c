static void avc_luma_hv_qrt_and_aver_dst_16x16_msa(const uint8_t *src_x,

                                                   const uint8_t *src_y,

                                                   int32_t src_stride,

                                                   uint8_t *dst,

                                                   int32_t dst_stride)

{

    uint32_t multiple8_cnt;



    for (multiple8_cnt = 2; multiple8_cnt--;) {

        avc_luma_hv_qrt_and_aver_dst_8x8_msa(src_x, src_y, src_stride,

                                             dst, dst_stride);



        src_x += 8;

        src_y += 8;

        dst += 8;

    }



    src_x += (8 * src_stride) - 16;

    src_y += (8 * src_stride) - 16;

    dst += (8 * dst_stride) - 16;



    for (multiple8_cnt = 2; multiple8_cnt--;) {

        avc_luma_hv_qrt_and_aver_dst_8x8_msa(src_x, src_y, src_stride,

                                             dst, dst_stride);



        src_x += 8;

        src_y += 8;

        dst += 8;

    }

}
