void ff_avg_h264_qpel4_mc00_msa(uint8_t *dst, const uint8_t *src,

                                ptrdiff_t stride)

{

    avg_width4_msa(src, stride, dst, stride, 4);

}
