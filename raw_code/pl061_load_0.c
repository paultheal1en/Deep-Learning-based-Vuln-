static int pl061_load(QEMUFile *f, void *opaque, int version_id)

{

    pl061_state *s = (pl061_state *)opaque;

    if (version_id != 1)

        return -EINVAL;



    s->locked = qemu_get_be32(f);

    s->data = qemu_get_be32(f);

    s->old_data = qemu_get_be32(f);

    s->dir = qemu_get_be32(f);

    s->isense = qemu_get_be32(f);

    s->ibe = qemu_get_be32(f);

    s->iev = qemu_get_be32(f);

    s->im = qemu_get_be32(f);

    s->istate = qemu_get_be32(f);

    s->afsel = qemu_get_be32(f);

    s->dr2r = qemu_get_be32(f);

    s->dr4r = qemu_get_be32(f);

    s->dr8r = qemu_get_be32(f);

    s->odr = qemu_get_be32(f);

    s->pur = qemu_get_be32(f);

    s->pdr = qemu_get_be32(f);

    s->slr = qemu_get_be32(f);

    s->den = qemu_get_be32(f);

    s->cr = qemu_get_be32(f);

    s->float_high = qemu_get_be32(f);



    return 0;

}
