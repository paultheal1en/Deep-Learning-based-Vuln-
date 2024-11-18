static int fd_open(BlockDriverState *bs)

{

    BDRVRawState *s = bs->opaque;

    int last_media_present;



    if (s->type != FTYPE_FD)

        return 0;

    last_media_present = (s->fd >= 0);

    if (s->fd >= 0 &&

        (qemu_get_clock(rt_clock) - s->fd_open_time) >= FD_OPEN_TIMEOUT) {

        close(s->fd);

        s->fd = -1;

        raw_close_fd_pool(s);

#ifdef DEBUG_FLOPPY

        printf("Floppy closed\n");

#endif

    }

    if (s->fd < 0) {

        if (s->fd_got_error &&

            (qemu_get_clock(rt_clock) - s->fd_error_time) < FD_OPEN_TIMEOUT) {

#ifdef DEBUG_FLOPPY

            printf("No floppy (open delayed)\n");

#endif

            return -EIO;

        }

        s->fd = open(bs->filename, s->fd_open_flags);

        if (s->fd < 0) {

            s->fd_error_time = qemu_get_clock(rt_clock);

            s->fd_got_error = 1;

            if (last_media_present)

                s->fd_media_changed = 1;

#ifdef DEBUG_FLOPPY

            printf("No floppy\n");

#endif

            return -EIO;

        }

#ifdef DEBUG_FLOPPY

        printf("Floppy opened\n");

#endif

    }

    if (!last_media_present)

        s->fd_media_changed = 1;

    s->fd_open_time = qemu_get_clock(rt_clock);

    s->fd_got_error = 0;

    return 0;

}
