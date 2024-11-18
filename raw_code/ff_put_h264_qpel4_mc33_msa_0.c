void ff_put_h264_qpel4_mc33_msa(uint8_t *dst, const uint8_t *src,

                                ptrdiff_t stride)

{

    avc_luma_hv_qrt_4w_msa(src + stride - 2,

                           src - (stride * 2) +

                           sizeof(uint8_t), stride, dst, stride, 4);

}
