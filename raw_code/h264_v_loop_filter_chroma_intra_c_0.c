static void h264_v_loop_filter_chroma_intra_c(uint8_t *pix, int stride, int alpha, int beta)

{

    h264_loop_filter_chroma_intra_c(pix, stride, 1, alpha, beta);

}
