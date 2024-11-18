static void free_schro_frame(SchroFrame *frame, void *priv)

{

    AVFrame *p_pic = priv;

    av_frame_free(&p_pic);

}
