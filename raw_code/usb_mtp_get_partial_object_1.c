static MTPData *usb_mtp_get_partial_object(MTPState *s, MTPControl *c,

                                           MTPObject *o)

{

    MTPData *d = usb_mtp_data_alloc(c);

    off_t offset;



    trace_usb_mtp_op_get_partial_object(s->dev.addr, o->handle, o->path,

                                        c->argv[1], c->argv[2]);



    d->fd = open(o->path, O_RDONLY);

    if (d->fd == -1) {


        return NULL;

    }



    offset = c->argv[1];

    if (offset > o->stat.st_size) {

        offset = o->stat.st_size;

    }

    lseek(d->fd, offset, SEEK_SET);



    d->length = c->argv[2];

    if (d->length > o->stat.st_size - offset) {

        d->length = o->stat.st_size - offset;

    }



    return d;

}