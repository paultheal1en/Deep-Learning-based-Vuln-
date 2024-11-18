static void hscroll(AVCodecContext *avctx)

{

    AnsiContext *s = avctx->priv_data;

    int i;



    if (s->y < avctx->height - s->font_height) {

        s->y += s->font_height;

        return;

    }



    i = 0;

    for (; i < avctx->height - s->font_height; i++)

        memcpy(s->frame->data[0] + i * s->frame->linesize[0],

               s->frame->data[0] + (i + s->font_height) * s->frame->linesize[0],

               avctx->width);

    for (; i < avctx->height; i++)

        memset(s->frame->data[0] + i * s->frame->linesize[0],

            DEFAULT_BG_COLOR, avctx->width);

}
