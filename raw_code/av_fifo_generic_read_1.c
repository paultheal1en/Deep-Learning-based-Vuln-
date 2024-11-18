int av_fifo_generic_read(AVFifoBuffer *f, int buf_size, void (*func)(void*, void*, int), void* dest)

{

    int size = av_fifo_size(f);



    if (size < buf_size)

        return -1;

    do {

        int len = FFMIN(f->end - f->rptr, buf_size);

        if(func) func(dest, f->rptr, len);

        else{

            memcpy(dest, f->rptr, len);

            dest = (uint8_t*)dest + len;

        }

        av_fifo_drain(f, len);

        buf_size -= len;

    } while (buf_size > 0);

    return 0;

}
