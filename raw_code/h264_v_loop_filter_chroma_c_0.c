static void h264_v_loop_filter_chroma_c(uint8_t *pix, int stride, int alpha, int beta, int8_t *tc0)

{

    h264_loop_filter_chroma_c(pix, stride, 1, alpha, beta, tc0);

}
