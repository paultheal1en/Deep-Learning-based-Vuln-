static inline int sub_left_prediction(HYuvContext *s, uint8_t *dst,

                                      const uint8_t *src, int w, int left)

{

    int i;

    if (s->bps <= 8) {

        if (w < 32) {

            for (i = 0; i < w; i++) {

                const int temp = src[i];

                dst[i] = temp - left;

                left   = temp;

            }

            return left;

        } else {

            for (i = 0; i < 32; i++) {

                const int temp = src[i];

                dst[i] = temp - left;

                left   = temp;

            }

            s->llvidencdsp.diff_bytes(dst + 32, src + 32, src + 31, w - 32);

            return src[w-1];

        }

    } else {

        const uint16_t *src16 = (const uint16_t *)src;

        uint16_t       *dst16 = (      uint16_t *)dst;

        if (w < 32) {

            for (i = 0; i < w; i++) {

                const int temp = src16[i];

                dst16[i] = temp - left;

                left   = temp;

            }

            return left;

        } else {

            for (i = 0; i < 16; i++) {

                const int temp = src16[i];

                dst16[i] = temp - left;

                left   = temp;

            }

            s->hencdsp.diff_int16(dst16 + 16, src16 + 16, src16 + 15, s->n - 1, w - 16);

            return src16[w-1];

        }

    }

}
