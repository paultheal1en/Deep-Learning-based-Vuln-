static int decode_rle_bpp2(AVCodecContext *avctx, AVFrame *p, GetByteContext *gbc)

{

    int offset = avctx->width;

    uint8_t *outdata = p->data[0];

    int i, j;



    for (i = 0; i < avctx->height; i++) {

        int size, left, code, pix;

        uint8_t *out = outdata;

        int pos = 0;



        /* size of packed line */

        size = left = bytestream2_get_be16(gbc);

        if (bytestream2_get_bytes_left(gbc) < size)

            return AVERROR_INVALIDDATA;



        /* decode line */

        while (left > 0) {

            code = bytestream2_get_byte(gbc);

            if (code & 0x80 ) { /* run */

                pix = bytestream2_get_byte(gbc);

                for (j = 0; j < 257 - code; j++) {

                    if (pos < offset)

                        out[pos++] = (pix & 0xC0) >> 6;

                    if (pos < offset)

                        out[pos++] = (pix & 0x30) >> 4;

                    if (pos < offset)

                        out[pos++] = (pix & 0x0C) >> 2;

                    if (pos < offset)

                        out[pos++] = (pix & 0x03);

                }

                left  -= 2;

            } else { /* copy */

                for (j = 0; j < code + 1; j++) {

                    pix = bytestream2_get_byte(gbc);

                    if (pos < offset)

                        out[pos++] = (pix & 0xC0) >> 6;

                    if (pos < offset)

                        out[pos++] = (pix & 0x30) >> 4;

                    if (pos < offset)

                        out[pos++] = (pix & 0x0C) >> 2;

                    if (pos < offset)

                        out[pos++] = (pix & 0x03);

                }

                left  -= 1 + (code + 1);

            }

        }

        outdata += p->linesize[0];

    }

    return 0;

}
