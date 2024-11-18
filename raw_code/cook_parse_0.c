static int cook_parse(AVCodecParserContext *s1, AVCodecContext *avctx,

                      const uint8_t **poutbuf, int *poutbuf_size,

                      const uint8_t *buf, int buf_size)

{

    CookParseContext *s = s1->priv_data;



    if (s->duration)

        s1->duration = s->duration;

    else if (avctx->extradata && avctx->extradata_size >= 8 && avctx->channels)

        s->duration = AV_RB16(avctx->extradata + 4) / avctx->channels;



    /* always return the full packet. this parser isn't doing any splitting or

       combining, only setting packet duration */

    *poutbuf      = buf;

    *poutbuf_size = buf_size;

    return buf_size;

}
