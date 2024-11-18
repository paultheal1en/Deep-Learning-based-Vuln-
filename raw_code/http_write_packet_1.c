static void http_write_packet(void *opaque, 

                              unsigned char *buf, int size)

{

    HTTPContext *c = opaque;



    if (c->buffer_ptr == c->buffer_end || !c->buffer_ptr)

        c->buffer_ptr = c->buffer_end = c->buffer;



    if (c->buffer_end - c->buffer + size > IOBUFFER_MAX_SIZE)

        abort();



    memcpy(c->buffer_end, buf, size);

    c->buffer_end += size;

}
