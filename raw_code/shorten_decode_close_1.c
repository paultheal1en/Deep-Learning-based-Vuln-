static av_cold int shorten_decode_close(AVCodecContext *avctx)

{

    ShortenContext *s = avctx->priv_data;

    int i;



    for (i = 0; i < s->channels; i++) {

        s->decoded[i] -= s->nwrap;

        av_freep(&s->decoded[i]);

        av_freep(&s->offset[i]);

    }

    av_freep(&s->bitstream);

    av_freep(&s->coeffs);



    return 0;

}
