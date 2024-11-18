void ff_vp56_init_range_decoder(VP56RangeCoder *c, const uint8_t *buf, int buf_size)

{

    c->high = 255;

    c->bits = -16;

    c->buffer = buf;

    c->end = buf + buf_size;

    c->code_word = bytestream_get_be24(&c->buffer);

}
