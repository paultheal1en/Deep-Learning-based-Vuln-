static inline int op(uint8_t **dst, const uint8_t *dst_end,

                     const uint8_t **buf, const uint8_t *buf_end,

                     int pixel, int count,

                     int *x, int width, int linesize)

{

    int remaining = width - *x;

    while(count > 0) {

        int striplen = FFMIN(count, remaining);

        if (buf) {

            striplen = FFMIN(striplen, buf_end - *buf);

            if (*buf >= buf_end)

                goto exhausted;

            memcpy(*dst, *buf, striplen);

            *buf += striplen;

        } else if (pixel >= 0)

            memset(*dst, pixel, striplen);

        *dst      += striplen;

        remaining -= striplen;

        count     -= striplen;

        if (remaining <= 0) {

            *dst      += linesize - width;

            remaining  = width;

        }

        if (linesize > 0) {

            if (*dst >= dst_end) goto exhausted;

        } else {

            if (*dst <= dst_end) goto exhausted;

        }

    }

    *x = width - remaining;

    return 0;



exhausted:

    *x = width - remaining;

    return 1;

}
