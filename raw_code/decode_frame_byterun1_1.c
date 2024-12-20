static int decode_frame_byterun1(AVCodecContext *avctx,

                            void *data, int *data_size,

                            AVPacket *avpkt)

{

    IffContext *s = avctx->priv_data;

    const uint8_t *buf = avpkt->size >= 2 ? avpkt->data + AV_RB16(avpkt->data) : NULL;

    const int buf_size = avpkt->size >= 2 ? avpkt->size - AV_RB16(avpkt->data) : 0;

    const uint8_t *buf_end = buf+buf_size;

    int y, plane, res;



    if ((res = extract_header(avctx, avpkt)) < 0)

        return res;

    if (s->init) {

        if ((res = avctx->reget_buffer(avctx, &s->frame)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "reget_buffer() failed\n");

            return res;

        }

    } else if ((res = avctx->get_buffer(avctx, &s->frame)) < 0) {

        av_log(avctx, AV_LOG_ERROR, "get_buffer() failed\n");

        return res;

    } else if (avctx->bits_per_coded_sample <= 8 && avctx->pix_fmt != PIX_FMT_GRAY8) {

        if ((res = ff_cmap_read_palette(avctx, (uint32_t*)s->frame.data[1])) < 0)

            return res;

    }

    s->init = 1;



    if (avctx->codec_tag == MKTAG('I','L','B','M')) { //interleaved

        if (avctx->pix_fmt == PIX_FMT_PAL8 || avctx->pix_fmt == PIX_FMT_GRAY8) {

            for(y = 0; y < avctx->height ; y++ ) {

                uint8_t *row = &s->frame.data[0][ y*s->frame.linesize[0] ];

                memset(row, 0, avctx->width);

                for (plane = 0; plane < s->bpp; plane++) {

                    buf += decode_byterun(s->planebuf, s->planesize, buf, buf_end);

                    decodeplane8(row, s->planebuf, s->planesize, plane);

                }

            }

        } else if (s->ham) { // HAM to PIX_FMT_BGR32

            for (y = 0; y < avctx->height ; y++) {

                uint8_t *row = &s->frame.data[0][y*s->frame.linesize[0]];

                memset(s->ham_buf, 0, avctx->width);

                for (plane = 0; plane < s->bpp; plane++) {

                    buf += decode_byterun(s->planebuf, s->planesize, buf, buf_end);

                    decodeplane8(s->ham_buf, s->planebuf, s->planesize, plane);

                }

                decode_ham_plane32((uint32_t *) row, s->ham_buf, s->ham_palbuf, s->planesize);

            }

        } else { //PIX_FMT_BGR32

            for(y = 0; y < avctx->height ; y++ ) {

                uint8_t *row = &s->frame.data[0][y*s->frame.linesize[0]];

                memset(row, 0, avctx->width << 2);

                for (plane = 0; plane < s->bpp; plane++) {

                    buf += decode_byterun(s->planebuf, s->planesize, buf, buf_end);

                    decodeplane32((uint32_t *) row, s->planebuf, s->planesize, plane);

                }

            }

        }

    } else if (avctx->pix_fmt == PIX_FMT_PAL8 || avctx->pix_fmt == PIX_FMT_GRAY8) { // IFF-PBM

        for(y = 0; y < avctx->height ; y++ ) {

            uint8_t *row = &s->frame.data[0][y*s->frame.linesize[0]];

            buf += decode_byterun(row, avctx->width, buf, buf_end);

        }

    } else { // IFF-PBM: HAM to PIX_FMT_BGR32

        for (y = 0; y < avctx->height ; y++) {

            uint8_t *row = &s->frame.data[0][y*s->frame.linesize[0]];

            buf += decode_byterun(s->ham_buf, avctx->width, buf, buf_end);

            decode_ham_plane32((uint32_t *) row, s->ham_buf, s->ham_palbuf, avctx->width);

        }

    }



    *data_size = sizeof(AVFrame);

    *(AVFrame*)data = s->frame;

    return buf_size;

}
