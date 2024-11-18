static void sunrast_image_write_image(AVCodecContext *avctx,

                                      const uint8_t *pixels,

                                      const uint32_t *palette_data,

                                      int linesize)

{

    SUNRASTContext *s = avctx->priv_data;

    const uint8_t *ptr;

    int len, alen, x;



    if (s->maplength) {     // palettized

        PutByteContext pb_r, pb_g;

        int len = s->maplength / 3;



        pb_r = s->p;

        bytestream2_skip_p(&s->p, len);

        pb_g = s->p;

        bytestream2_skip_p(&s->p, len);



        for (x = 0; x < len; x++) {

            uint32_t pixel = palette_data[x];



            bytestream2_put_byteu(&pb_r, (pixel >> 16) & 0xFF);

            bytestream2_put_byteu(&pb_g, (pixel >> 8)  & 0xFF);

            bytestream2_put_byteu(&s->p,  pixel        & 0xFF);

        }

    }



    len  = (s->depth * avctx->width + 7) >> 3;

    alen = len + (len & 1);

    ptr  = pixels;



     if (s->type == RT_BYTE_ENCODED) {

        uint8_t value, value2;

        int run;

        const uint8_t *end = pixels + avctx->height * linesize;



        ptr = pixels;



#define GET_VALUE ptr >= end ? 0 : x >= len ? ptr[len-1] : ptr[x]



        x = 0;

        value2 = GET_VALUE;

        while (ptr < end) {

            run = 1;

            value = value2;

            x++;

            if (x >= alen) {

                x = 0;

                ptr += linesize;

            }



            value2 = GET_VALUE;

            while (value2 == value && run < 256 && ptr < end) {

                x++;

                run++;

                if (x >= alen) {

                    x = 0;

                    ptr += linesize;

                }

                value2 = GET_VALUE;

            }



            if (run > 2 || value == RLE_TRIGGER) {

                bytestream2_put_byteu(&s->p, RLE_TRIGGER);

                bytestream2_put_byteu(&s->p, run - 1);

                if (run > 1)

                    bytestream2_put_byteu(&s->p, value);

            } else if (run == 1) {

                bytestream2_put_byteu(&s->p, value);

            } else

                bytestream2_put_be16u(&s->p, (value << 8) | value);

        }



        // update data length for header

        s->length = bytestream2_tell_p(&s->p) - 32 - s->maplength;

    } else {

        int y;

        for (y = 0; y < avctx->height; y++) {

            bytestream2_put_buffer(&s->p, ptr, len);

            if (len < alen)

                bytestream2_put_byteu(&s->p, 0);

            ptr += linesize;

        }

    }

}
