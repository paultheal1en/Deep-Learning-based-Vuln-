static int decode_frame_ilbm(AVCodecContext *avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    IffContext *s = avctx->priv_data;

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    const uint8_t *buf_end = buf+buf_size;

    int y, plane;



    if (avctx->reget_buffer(avctx, &s->frame) < 0){

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return -1;

    }



    for(y = 0; y < avctx->height; y++ ) {

        uint8_t *row = &s->frame.data[0][ y*s->frame.linesize[0] ];

        memset(row, 0, avctx->pix_fmt == PIX_FMT_PAL8 ? avctx->width : (avctx->width * 4));

        for (plane = 0; plane < avctx->bits_per_coded_sample && buf < buf_end; plane++) {

            if (avctx->pix_fmt == PIX_FMT_PAL8) {

                decodeplane8(row, buf, FFMIN(s->planesize, buf_end - buf), avctx->bits_per_coded_sample, plane);

            } else { // PIX_FMT_BGR32

                decodeplane32(row, buf, FFMIN(s->planesize, buf_end - buf), avctx->bits_per_coded_sample, plane);

            }

            buf += s->planesize;

        }

    }



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = s->frame;

    return buf_size;

}
