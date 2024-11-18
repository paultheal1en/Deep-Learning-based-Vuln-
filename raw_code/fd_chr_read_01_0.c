static void fd_chr_read(void *opaque)

{

    CharDriverState *chr = opaque;

    FDCharDriver *s = chr->opaque;

    int size, len;

    uint8_t buf[1024];



    len = sizeof(buf);

    if (len > s->max_size)

        len = s->max_size;

    if (len == 0)

        return;

    size = read(s->fd_in, buf, len);

    if (size == 0) {

        /* FD has been closed. Remove it from the active list.  */

        qemu_set_fd_handler2(s->fd_in, NULL, NULL, NULL, NULL);

        qemu_chr_event(chr, CHR_EVENT_CLOSED);

        return;

    }

    if (size > 0) {

        qemu_chr_read(chr, buf, size);

    }

}
