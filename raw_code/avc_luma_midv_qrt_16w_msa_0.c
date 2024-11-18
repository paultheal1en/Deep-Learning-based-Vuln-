static void avc_luma_midv_qrt_16w_msa(const uint8_t *src, int32_t src_stride,

                                      uint8_t *dst, int32_t dst_stride,

                                      int32_t height, uint8_t vert_offset)

{

    uint32_t multiple8_cnt;



    for (multiple8_cnt = 2; multiple8_cnt--;) {

        avc_luma_midv_qrt_8w_msa(src, src_stride, dst, dst_stride, height,

                                 vert_offset);



        src += 8;

        dst += 8;

    }

}
