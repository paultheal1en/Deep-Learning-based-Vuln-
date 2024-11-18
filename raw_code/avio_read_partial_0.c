int avio_read_partial(AVIOContext *s, unsigned char *buf, int size)

{

    int len;



    if (size < 0)

        return -1;



    if (s->read_packet && s->write_flag) {

        len = s->read_packet(s->opaque, buf, size);

        if (len > 0)

            s->pos += len;

        return len;

    }



    len = s->buf_end - s->buf_ptr;

    if (len == 0) {

        /* Reset the buf_end pointer to the start of the buffer, to make sure

         * the fill_buffer call tries to read as much data as fits into the

         * full buffer, instead of just what space is left after buf_end.

         * This avoids returning partial packets at the end of the buffer,

         * for packet based inputs.

         */

        s->buf_end = s->buf_ptr = s->buffer;

        fill_buffer(s);

        len = s->buf_end - s->buf_ptr;

    }

    if (len > size)

        len = size;

    memcpy(buf, s->buf_ptr, len);

    s->buf_ptr += len;

    if (!len) {

        if (s->error)      return s->error;

        if (avio_feof(s))  return AVERROR_EOF;

    }

    return len;

}
