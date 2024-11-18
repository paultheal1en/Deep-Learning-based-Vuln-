static int dvvideo_close(AVCodecContext *c)

{

    DVVideoContext *s = c->priv_data;



    av_free(s->dv_anchor);



    return 0;

}
