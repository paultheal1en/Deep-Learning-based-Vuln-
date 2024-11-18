static void dyn_buf_write(void *opaque, UINT8 *buf, int buf_size)

{

    DynBuffer *d = opaque;

    int new_size, new_allocated_size;

    UINT8 *new_buffer;

    

    /* reallocate buffer if needed */

    new_size = d->pos + buf_size;

    new_allocated_size = d->allocated_size;

    while (new_size > new_allocated_size) {

        if (!new_allocated_size)

            new_allocated_size = new_size;

        else

            new_allocated_size = (new_allocated_size * 3) / 2;

    }

    

    if (new_allocated_size > d->allocated_size) {

        new_buffer = av_malloc(new_allocated_size);

        if (!new_buffer)

            return;

        memcpy(new_buffer, d->buffer, d->size);

        av_free(d->buffer);

        d->buffer = new_buffer;

        d->allocated_size = new_allocated_size;

    }

    memcpy(d->buffer + d->pos, buf, buf_size);

    d->pos = new_size;

    if (d->pos > d->size)

        d->size = d->pos;

}
