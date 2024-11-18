void av_fifo_write(AVFifoBuffer *f, const uint8_t *buf, int size)

{

    while (size > 0) {

        int len = FFMIN(f->end - f->wptr, size);

        memcpy(f->wptr, buf, len);

        f->wptr += len;

        if (f->wptr >= f->end)

            f->wptr = f->buffer;

        buf += len;

        size -= len;

    }

}
