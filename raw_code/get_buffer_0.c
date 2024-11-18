int get_buffer(ByteIOContext *s, unsigned char *buf, int size)

{

    int len, size1;



    size1 = size;

    while (size > 0) {

        len = s->buf_end - s->buf_ptr;

        if (len > size)

            len = size;

        if (len == 0) {

            fill_buffer(s);

            len = s->buf_end - s->buf_ptr;

            if (len == 0)

                break;

        } else {

            memcpy(buf, s->buf_ptr, len);

            buf += len;

            s->buf_ptr += len;

            size -= len;

        }

    }

    return size1 - size;

}
