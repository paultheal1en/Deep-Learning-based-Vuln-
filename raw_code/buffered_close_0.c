static int buffered_close(void *opaque)

{

    MigrationState *s = opaque;



    DPRINTF("closing\n");



    s->xfer_limit = INT_MAX;

    while (!qemu_file_get_error(s->file) && s->buffer_size) {

        buffered_flush(s);

    }

    return migrate_fd_close(s);

}
