static int amr_wb_encode_frame(AVCodecContext *avctx,

                            unsigned char *frame/*out*/, int buf_size, void *data/*in*/)

{

    AMRWBContext *s;

    int size;

    s = (AMRWBContext*) avctx->priv_data;

    s->mode=getWBBitrateMode(avctx->bit_rate);

    size = E_IF_encode(s->state, s->mode, data, frame, s->allow_dtx);

    return size;

}
