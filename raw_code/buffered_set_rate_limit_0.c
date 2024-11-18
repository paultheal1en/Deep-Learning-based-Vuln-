static size_t buffered_set_rate_limit(void *opaque, size_t new_rate)

{

    QEMUFileBuffered *s = opaque;



    if (s->has_error)

        goto out;



    s->xfer_limit = new_rate / 10;

    

out:

    return s->xfer_limit;

}
