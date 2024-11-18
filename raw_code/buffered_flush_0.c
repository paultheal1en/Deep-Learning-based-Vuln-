static void buffered_flush(QEMUFileBuffered *s)

{

    size_t offset = 0;

    int error;



    error = qemu_file_get_error(s->file);

    if (error != 0) {

        DPRINTF("flush when error, bailing: %s\n", strerror(-error));

        return;

    }



    DPRINTF("flushing %zu byte(s) of data\n", s->buffer_size);



    while (s->bytes_xfer < s->xfer_limit && offset < s->buffer_size) {

        ssize_t ret;



        ret = migrate_fd_put_buffer(s->migration_state, s->buffer + offset,

                                    s->buffer_size - offset);

        if (ret == -EAGAIN) {

            DPRINTF("backend not ready, freezing\n");

            s->freeze_output = 1;

            break;

        }



        if (ret <= 0) {

            DPRINTF("error flushing data, %zd\n", ret);

            qemu_file_set_error(s->file, ret);

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

}
