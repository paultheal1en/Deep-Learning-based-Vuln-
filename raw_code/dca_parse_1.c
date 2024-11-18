static int dca_parse(AVCodecParserContext *s, AVCodecContext *avctx,

                     const uint8_t **poutbuf, int *poutbuf_size,

                     const uint8_t *buf, int buf_size)

{

    DCAParseContext *pc1 = s->priv_data;

    ParseContext *pc = &pc1->pc;

    int next, duration, sample_rate;



    if (s->flags & PARSER_FLAG_COMPLETE_FRAMES) {

        next = buf_size;

    } else {

        next = dca_find_frame_end(pc1, buf, buf_size);



        if (ff_combine_frame(pc, next, &buf, &buf_size) < 0) {

            *poutbuf      = NULL;

            *poutbuf_size = 0;

            return buf_size;

        }

    }



    /* read the duration and sample rate from the frame header */

    if (!dca_parse_params(buf, buf_size, &duration, &sample_rate, &pc1->framesize)) {

        s->duration        = duration;

        avctx->sample_rate = sample_rate;

    } else

        s->duration = 0;



    *poutbuf      = buf;

    *poutbuf_size = buf_size;

    return next;

}
