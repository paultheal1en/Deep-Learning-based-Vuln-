static int g729_parse(AVCodecParserContext *s1, AVCodecContext *avctx,

                     const uint8_t **poutbuf, int *poutbuf_size,

                     const uint8_t *buf, int buf_size)

{

    G729ParseContext *s = s1->priv_data;

    ParseContext *pc = &s->pc;

    int next;



    if (!s->block_size) {

        switch (avctx->codec_id) {

        case AV_CODEC_ID_G729:

            /* FIXME: replace this heuristic block_size with more precise estimate */

            s->block_size = (avctx->bit_rate < 8000) ? G729D_6K4_BLOCK_SIZE : G729_8K_BLOCK_SIZE;

            s->duration   = avctx->frame_size;

            break;

        default:

            *poutbuf      = buf;

            *poutbuf_size = buf_size;

            av_log(avctx, AV_LOG_ERROR, "Invalid codec_id\n");

            return buf_size;

        }

    }



    if (!s->remaining)

        s->remaining = s->block_size;

    if (s->remaining <= buf_size) {

        next = s->remaining;

        s->remaining = 0;

    } else {

        next = END_NOT_FOUND;

        s->remaining -= buf_size;

    }



    if (ff_combine_frame(pc, next, &buf, &buf_size) < 0 || !buf_size) {

        *poutbuf      = NULL;

        *poutbuf_size = 0;

        return buf_size;

    }



    s1->duration = s->duration;



    *poutbuf      = buf;

    *poutbuf_size = buf_size;

    return next;

}
