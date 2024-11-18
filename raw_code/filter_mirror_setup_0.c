static void filter_mirror_setup(NetFilterState *nf, Error **errp)

{

    MirrorState *s = FILTER_MIRROR(nf);



    if (!s->outdev) {

        error_setg(errp, "filter filter mirror needs 'outdev' "

                   "property set");

        return;

    }



    s->chr_out = qemu_chr_find(s->outdev);

    if (s->chr_out == NULL) {

        error_set(errp, ERROR_CLASS_DEVICE_NOT_FOUND,

                  "Device '%s' not found", s->outdev);

        return;

    }



    if (qemu_chr_fe_claim(s->chr_out) != 0) {

        error_setg(errp, QERR_DEVICE_IN_USE, s->outdev);

        return;

    }

}
