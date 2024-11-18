static inline void copy(LZOContext *c, int cnt)

{

    register const uint8_t *src = c->in;

    register uint8_t *dst       = c->out;


    if (cnt > c->in_end - src) {

        cnt       = FFMAX(c->in_end - src, 0);

        c->error |= AV_LZO_INPUT_DEPLETED;

    }

    if (cnt > c->out_end - dst) {

        cnt       = FFMAX(c->out_end - dst, 0);

        c->error |= AV_LZO_OUTPUT_FULL;

    }

#if defined(INBUF_PADDED) && defined(OUTBUF_PADDED)

    AV_COPY32U(dst, src);

    src += 4;

    dst += 4;

    cnt -= 4;

    if (cnt > 0)

#endif

    memcpy(dst, src, cnt);

    c->in  = src + cnt;

    c->out = dst + cnt;

}