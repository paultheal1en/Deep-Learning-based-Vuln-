static int g726_decode_frame(AVCodecContext *avctx,

                             void *data, int *data_size,

                             AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    G726Context *c = avctx->priv_data;

    int16_t *samples = data;

    GetBitContext gb;



    init_get_bits(&gb, buf, buf_size * 8);



    while (get_bits_count(&gb) + c->code_size <= buf_size*8)

        *samples++ = g726_decode(c, get_bits(&gb, c->code_size));



    if(buf_size*8 != get_bits_count(&gb))

        av_log(avctx, AV_LOG_ERROR, "Frame invalidly split, missing parser?\n");



    *data_size = (uint8_t*)samples - (uint8_t*)data;

    return buf_size;

}
