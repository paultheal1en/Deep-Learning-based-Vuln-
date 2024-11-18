static int get_qcx(Jpeg2000DecoderContext *s, int n, Jpeg2000QuantStyle *q)

{

    int i, x;



    if (s->buf_end - s->buf < 1)

        return AVERROR_INVALIDDATA;



    x = bytestream_get_byte(&s->buf); // Sqcd



    q->nguardbits = x >> 5;

    q->quantsty   = x & 0x1f;



    if (q->quantsty == JPEG2000_QSTY_NONE) {

        n -= 3;

        if (s->buf_end - s->buf < n)

            return AVERROR_INVALIDDATA;

        for (i = 0; i < n; i++)

            q->expn[i] = bytestream_get_byte(&s->buf) >> 3;

    } else if (q->quantsty == JPEG2000_QSTY_SI) {

        if (s->buf_end - s->buf < 2)

            return AVERROR_INVALIDDATA;

        x          = bytestream_get_be16(&s->buf);

        q->expn[0] = x >> 11;

        q->mant[0] = x & 0x7ff;

        for (i = 1; i < JPEG2000_MAX_DECLEVELS * 3; i++) {

            int curexpn = FFMAX(0, q->expn[0] - (i - 1) / 3);

            q->expn[i] = curexpn;

            q->mant[i] = q->mant[0];

        }

    } else {

        n = (n - 3) >> 1;

        if (s->buf_end - s->buf < n)

            return AVERROR_INVALIDDATA;

        for (i = 0; i < n; i++) {

            x          = bytestream_get_be16(&s->buf);

            q->expn[i] = x >> 11;

            q->mant[i] = x & 0x7ff;

        }

    }

    return 0;

}
