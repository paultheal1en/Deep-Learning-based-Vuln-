static int64_t buffered_set_rate_limit(void *opaque, int64_t new_rate)

{

    QEMUFileBuffered *s = opaque;

    if (qemu_file_get_error(s->file)) {

        goto out;

    }

    if (new_rate > SIZE_MAX) {

        new_rate = SIZE_MAX;

    }



    s->xfer_limit = new_rate / 10;

    

out:

    return s->xfer_limit;

}
