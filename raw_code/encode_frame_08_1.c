static int encode_frame(AVCodecContext *avctx, unsigned char *buf,

                        int buf_size, void *data)

{

    const AVFrame *pic = data;

    int aligned_width = ((avctx->width + 47) / 48) * 48;

    int stride = aligned_width * 8 / 3;

    int h, w;

    const uint16_t *y = (const uint16_t*)pic->data[0];

    const uint16_t *u = (const uint16_t*)pic->data[1];

    const uint16_t *v = (const uint16_t*)pic->data[2];

    uint8_t *p = buf;

    uint8_t *pdst = buf;



    if (buf_size < aligned_width * avctx->height * 8 / 3) {

        av_log(avctx, AV_LOG_ERROR, "output buffer too small\n");

        return -1;

    }



#define CLIP(v) av_clip(v, 4, 1019)



#define WRITE_PIXELS(a, b, c)           \

    do {                                \

        val =   CLIP(*a++);             \

        val |= (CLIP(*b++) << 10) |     \

               (CLIP(*c++) << 20);      \

        bytestream_put_le32(&p, val);   \

    } while (0)



    for (h = 0; h < avctx->height; h++) {

        uint32_t val;

        for (w = 0; w < avctx->width - 5; w += 6) {

            WRITE_PIXELS(u, y, v);

            WRITE_PIXELS(y, u, y);

            WRITE_PIXELS(v, y, u);

            WRITE_PIXELS(y, v, y);

        }

        if (w < avctx->width - 1) {

            WRITE_PIXELS(u, y, v);



            val = CLIP(*y++);

            if (w == avctx->width - 2)

                bytestream_put_le32(&p, val);

        }

        if (w < avctx->width - 3) {

            val |= (CLIP(*u++) << 10) | (CLIP(*y++) << 20);

            bytestream_put_le32(&p, val);



            val = CLIP(*v++) | (CLIP(*y++) << 10);

            bytestream_put_le32(&p, val);

        }



        pdst += stride;

        memset(p, 0, pdst - p);

        p = pdst;

        y += pic->linesize[0] / 2 - avctx->width;

        u += pic->linesize[1] / 2 - avctx->width / 2;

        v += pic->linesize[2] / 2 - avctx->width / 2;

    }



    return p - buf;

}
