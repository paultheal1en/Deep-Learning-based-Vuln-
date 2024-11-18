static int serial_load(QEMUFile *f, void *opaque, int version_id)

{

    SerialState *s = opaque;



    if(version_id > 2)

        return -EINVAL;



    if (version_id >= 2)

        qemu_get_be16s(f, &s->divider);

    else

        s->divider = qemu_get_byte(f);

    qemu_get_8s(f,&s->rbr);

    qemu_get_8s(f,&s->ier);

    qemu_get_8s(f,&s->iir);

    qemu_get_8s(f,&s->lcr);

    qemu_get_8s(f,&s->mcr);

    qemu_get_8s(f,&s->lsr);

    qemu_get_8s(f,&s->msr);

    qemu_get_8s(f,&s->scr);



    return 0;

}
