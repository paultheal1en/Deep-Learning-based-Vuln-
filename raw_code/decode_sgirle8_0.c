static int decode_sgirle8(AVCodecContext *avctx, uint8_t *dst, const uint8_t *src, int src_size, int width, int height, int linesize)

{

    const uint8_t *src_end = src + src_size;

    int x = 0, y = 0;



#define INC_XY(n) \

    x += n; \

    if (x >= width) { \

        y++; \

        if (y >= height) \

            return 0; \

        x = 0; \

    }



    while (src_end - src >= 2) {

        uint8_t v = *src++;

        if (v > 0 && v < 0xC0) {

            do {

                int length = FFMIN(v, width - x);

                memset(dst + y*linesize + x, RGB332_TO_BGR8(*src), length);

                INC_XY(length);

                v   -= length;

            } while (v > 0);

            src++;

        } else if (v >= 0xC1) {

            v -= 0xC0;

            do {

                int length = FFMIN3(v, width - x, src_end - src);

                if (src_end - src < length)

                    break;

                memcpy_rgb332_to_bgr8(dst + y*linesize + x, src, length);

                INC_XY(length);

                src += length;

                v   -= length;

            } while (v > 0);

        } else {

            avpriv_request_sample(avctx, "opcode %d", v);

            return AVERROR_PATCHWELCOME;

        }

    }

    return 0;

}
