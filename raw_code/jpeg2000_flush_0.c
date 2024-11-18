static void jpeg2000_flush(Jpeg2000DecoderContext *s)

{

    if (*s->buf == 0xff)

        s->buf++;

    s->bit_index = 8;

    s->buf++;

}
