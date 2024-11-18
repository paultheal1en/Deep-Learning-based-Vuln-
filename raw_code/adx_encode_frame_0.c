static int adx_encode_frame(AVCodecContext *avctx, uint8_t *frame,

                            int buf_size, void *data)

{

    ADXContext *c          = avctx->priv_data;

    const int16_t *samples = data;

    uint8_t *dst           = frame;

    int ch;



    if (!c->header_parsed) {

        int hdrsize = adx_encode_header(avctx, dst, buf_size);

        dst += hdrsize;

        c->header_parsed = 1;

    }



    for (ch = 0; ch < avctx->channels; ch++) {

        adx_encode(c, dst, samples + ch, &c->prev[ch], avctx->channels);

        dst += BLOCK_SIZE;

    }

    return dst - frame;

}
