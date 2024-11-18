void ff_avg_h264_qpel16_mc00_msa(uint8_t *dst, const uint8_t *src,

                                 ptrdiff_t stride)

{

    avg_width16_msa(src, stride, dst, stride, 16);

}
