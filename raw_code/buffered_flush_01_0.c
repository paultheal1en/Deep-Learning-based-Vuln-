static ssize_t buffered_flush(QEMUFileBuffered *s)

{

    size_t offset = 0;

    ssize_t ret = 0;



    DPRINTF("flushing %zu byte(s) of data\n", s->buffer_size);



    while (s->bytes_xfer < s->xfer_limit && offset < s->buffer_size) {

        size_t to_send = MIN(s->buffer_size - offset, s->xfer_limit - s->bytes_xfer);

        ret = migrate_fd_put_buffer(s->migration_state, s->buffer + offset,

                                    to_send);

        if (ret <= 0) {

            DPRINTF("error flushing data, %zd\n", ret);

            break;

        } else {

            DPRINTF("flushed %zd byte(s)\n", ret);

            offset += ret;

            s->bytes_xfer += ret;

        }

    }



    DPRINTF("flushed %zu of %zu byte(s)\n", offset, s->buffer_size);

    memmove(s->buffer, s->buffer + offset, s->buffer_size - offset);

    s->buffer_size -= offset;



    if (ret < 0) {

        return ret;

    }

    return offset;

}
