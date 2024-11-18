static int adx_parse(AVCodecParserContext *s1,

                           AVCodecContext *avctx,

                           const uint8_t **poutbuf, int *poutbuf_size,

                           const uint8_t *buf, int buf_size)

{

    ADXParseContext *s = s1->priv_data;

    ParseContext *pc = &s->pc;

    int next = END_NOT_FOUND;



    if (!avctx->extradata_size) {

        int ret;



        ff_combine_frame(pc, END_NOT_FOUND, &buf, &buf_size);



        if (!s->header_size && pc->index >= MIN_HEADER_SIZE) {

            if (ret = avpriv_adx_decode_header(avctx, pc->buffer, pc->index,

                                               &s->header_size, NULL))

                return AVERROR_INVALIDDATA;

            s->block_size = BLOCK_SIZE * avctx->channels;

        }

        if (s->header_size && s->header_size <= pc->index) {

            avctx->extradata = av_mallocz(s->header_size + FF_INPUT_BUFFER_PADDING_SIZE);

            if (!avctx->extradata)

                return AVERROR(ENOMEM);

            avctx->extradata_size = s->header_size;

            memcpy(avctx->extradata, pc->buffer, s->header_size);

            memmove(pc->buffer, pc->buffer + s->header_size, s->header_size);

            pc->index -= s->header_size;

        }

        *poutbuf      = NULL;

        *poutbuf_size = 0;

        return buf_size;

    }



    if (pc->index - s->buf_pos >= s->block_size) {

        *poutbuf      = &pc->buffer[s->buf_pos];

        *poutbuf_size = s->block_size;

        s->buf_pos   += s->block_size;

        return 0;

    }

    if (pc->index && s->buf_pos) {

        memmove(pc->buffer, &pc->buffer[s->buf_pos], pc->index - s->buf_pos);

        pc->index -= s->buf_pos;

        s->buf_pos = 0;

    }

    if (buf_size + pc->index >= s->block_size)

        next = s->block_size - pc->index;



    if (ff_combine_frame(pc, next, &buf, &buf_size) < 0 || !buf_size) {

        *poutbuf      = NULL;

        *poutbuf_size = 0;

        return buf_size;

    }

    *poutbuf = buf;

    *poutbuf_size = buf_size;

    return next;

}
