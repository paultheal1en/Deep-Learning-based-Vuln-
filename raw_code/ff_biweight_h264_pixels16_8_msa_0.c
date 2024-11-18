void ff_biweight_h264_pixels16_8_msa(uint8_t *dst, uint8_t *src,

                                     int stride, int height,

                                     int log2_denom, int weight_dst,

                                     int weight_src, int offset)

{

    avc_biwgt_16width_msa(src, stride,

                          dst, stride,

                          height, log2_denom,

                          weight_src, weight_dst, offset);

}
