static int pcx_encode_frame(AVCodecContext *avctx,

                            unsigned char *buf, int buf_size, void *data)

{

    PCXContext *s = avctx->priv_data;

    AVFrame *const pict = &s->picture;

    const uint8_t *buf_start = buf;

    const uint8_t *buf_end   = buf + buf_size;



    int bpp, nplanes, i, y, line_bytes, written;

    const uint32_t *pal = NULL;


    const uint8_t *src;



    *pict = *(AVFrame *)data;

    pict->pict_type = AV_PICTURE_TYPE_I;

    pict->key_frame = 1;



    if (avctx->width > 65535 || avctx->height > 65535) {

        av_log(avctx, AV_LOG_ERROR, "image dimensions do not fit in 16 bits\n");

        return -1;

    }



    switch (avctx->pix_fmt) {

    case PIX_FMT_RGB24:

        bpp = 8;

        nplanes = 3;

        break;

    case PIX_FMT_RGB8:

    case PIX_FMT_BGR8:

    case PIX_FMT_RGB4_BYTE:

    case PIX_FMT_BGR4_BYTE:

    case PIX_FMT_GRAY8:

        bpp = 8;

        nplanes = 1;

        ff_set_systematic_pal2(palette256, avctx->pix_fmt);

        pal = palette256;

        break;

    case PIX_FMT_PAL8:

        bpp = 8;

        nplanes = 1;

        pal = (uint32_t *)pict->data[1];

        break;

    case PIX_FMT_MONOBLACK:

        bpp = 1;

        nplanes = 1;

        pal = monoblack_pal;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "unsupported pixfmt\n");

        return -1;

    }



    line_bytes = (avctx->width * bpp + 7) >> 3;

    line_bytes = (line_bytes + 1) & ~1;



    bytestream_put_byte(&buf, 10);                  // manufacturer

    bytestream_put_byte(&buf, 5);                   // version

    bytestream_put_byte(&buf, 1);                   // encoding

    bytestream_put_byte(&buf, bpp);                 // bits per pixel per plane

    bytestream_put_le16(&buf, 0);                   // x min

    bytestream_put_le16(&buf, 0);                   // y min

    bytestream_put_le16(&buf, avctx->width - 1);    // x max

    bytestream_put_le16(&buf, avctx->height - 1);   // y max

    bytestream_put_le16(&buf, 0);                   // horizontal DPI

    bytestream_put_le16(&buf, 0);                   // vertical DPI

    for (i = 0; i < 16; i++)

        bytestream_put_be24(&buf, pal ? pal[i] : 0);// palette (<= 16 color only)

    bytestream_put_byte(&buf, 0);                   // reserved

    bytestream_put_byte(&buf, nplanes);             // number of planes

    bytestream_put_le16(&buf, line_bytes);          // scanline plane size in bytes



    while (buf - buf_start < 128)

        *buf++= 0;



    src = pict->data[0];



    for (y = 0; y < avctx->height; y++) {

        if ((written = pcx_rle_encode(buf, buf_end - buf,

                                      src, line_bytes, nplanes)) < 0) {

            av_log(avctx, AV_LOG_ERROR, "buffer too small\n");

            return -1;

        }

        buf += written;

        src += pict->linesize[0];

    }



    if (nplanes == 1 && bpp == 8) {

        if (buf_end - buf < 257) {

            av_log(avctx, AV_LOG_ERROR, "buffer too small\n");

            return -1;

        }

        bytestream_put_byte(&buf, 12);

        for (i = 0; i < 256; i++) {

            bytestream_put_be24(&buf, pal[i]);

        }

    }



    return buf - buf_start;

}