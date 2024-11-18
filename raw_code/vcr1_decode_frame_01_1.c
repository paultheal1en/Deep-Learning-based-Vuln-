static int vcr1_decode_frame(AVCodecContext *avctx, void *data,

                             int *got_frame, AVPacket *avpkt)

{

    const uint8_t *buf        = avpkt->data;

    int buf_size              = avpkt->size;

    VCR1Context *const a      = avctx->priv_data;

    AVFrame *const p          = data;

    const uint8_t *bytestream = buf;

    int i, x, y, ret;



    if ((ret = ff_get_buffer(avctx, p, 0)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return ret;

    }

    p->pict_type = AV_PICTURE_TYPE_I;

    p->key_frame = 1;



    if (buf_size < 32)

        goto packet_small;



    for (i = 0; i < 16; i++) {

        a->delta[i] = *bytestream++;

        bytestream++;

        buf_size--;

    }



    for (y = 0; y < avctx->height; y++) {

        int offset;

        uint8_t *luma = &p->data[0][y * p->linesize[0]];



        if ((y & 3) == 0) {

            uint8_t *cb = &p->data[1][(y >> 2) * p->linesize[1]];

            uint8_t *cr = &p->data[2][(y >> 2) * p->linesize[2]];



            if (buf_size < 4 + avctx->width)

                goto packet_small;



            for (i = 0; i < 4; i++)

                a->offset[i] = *bytestream++;

            buf_size -= 4;



            offset = a->offset[0] - a->delta[bytestream[2] & 0xF];

            for (x = 0; x < avctx->width; x += 4) {

                luma[0]     = offset += a->delta[bytestream[2] & 0xF];

                luma[1]     = offset += a->delta[bytestream[2] >>  4];

                luma[2]     = offset += a->delta[bytestream[0] & 0xF];

                luma[3]     = offset += a->delta[bytestream[0] >>  4];

                luma       += 4;



                *cb++       = bytestream[3];

                *cr++       = bytestream[1];



                bytestream += 4;


            }

        } else {

            if (buf_size < avctx->width / 2)

                goto packet_small;



            offset = a->offset[y & 3] - a->delta[bytestream[2] & 0xF];



            for (x = 0; x < avctx->width; x += 8) {

                luma[0]     = offset += a->delta[bytestream[2] & 0xF];

                luma[1]     = offset += a->delta[bytestream[2] >>  4];

                luma[2]     = offset += a->delta[bytestream[3] & 0xF];

                luma[3]     = offset += a->delta[bytestream[3] >>  4];

                luma[4]     = offset += a->delta[bytestream[0] & 0xF];

                luma[5]     = offset += a->delta[bytestream[0] >>  4];

                luma[6]     = offset += a->delta[bytestream[1] & 0xF];

                luma[7]     = offset += a->delta[bytestream[1] >>  4];

                luma       += 8;

                bytestream += 4;


            }

        }

    }



    *got_frame = 1;



    return buf_size;

packet_small:

    av_log(avctx, AV_LOG_ERROR, "Input packet too small.\n");

    return AVERROR_INVALIDDATA;

}