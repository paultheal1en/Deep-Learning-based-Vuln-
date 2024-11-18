static void free_buffers(AVCodecContext *avctx)

{

    CFHDContext *s = avctx->priv_data;

    int i;



    for (i = 0; i < 4; i++) {

        av_freep(&s->plane[i].idwt_buf);

        av_freep(&s->plane[i].idwt_tmp);

    }

    s->a_height = 0;

    s->a_width  = 0;

}
