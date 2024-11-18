static int ram_decompress_buf(RamDecompressState *s, uint8_t *buf, int len)

{

    int ret, clen;



    s->zstream.avail_out = len;

    s->zstream.next_out = buf;

    while (s->zstream.avail_out > 0) {

        if (s->zstream.avail_in == 0) {

            if (qemu_get_be16(s->f) != RAM_CBLOCK_MAGIC)

                return -1;

            clen = qemu_get_be16(s->f);

            if (clen > IOBUF_SIZE)

                return -1;

            qemu_get_buffer(s->f, s->buf, clen);

            s->zstream.avail_in = clen;

            s->zstream.next_in = s->buf;

        }

        ret = inflate(&s->zstream, Z_PARTIAL_FLUSH);

        if (ret != Z_OK && ret != Z_STREAM_END) {

            return -1;

        }

    }

    return 0;

}
