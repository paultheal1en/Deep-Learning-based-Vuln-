static void avc_biwgt_4width_msa(uint8_t *src,

                                 int32_t src_stride,

                                 uint8_t *dst,

                                 int32_t dst_stride,

                                 int32_t height,

                                 int32_t log2_denom,

                                 int32_t src_weight,

                                 int32_t dst_weight,

                                 int32_t offset_in)

{

    if (2 == height) {

        avc_biwgt_4x2_msa(src, src_stride, dst, dst_stride,

                          log2_denom, src_weight, dst_weight,

                          offset_in);

    } else {

        avc_biwgt_4x4multiple_msa(src, src_stride, dst, dst_stride,

                                  height, log2_denom, src_weight,

                                  dst_weight, offset_in);

    }

}
