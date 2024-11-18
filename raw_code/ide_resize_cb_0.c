static void ide_resize_cb(void *opaque)

{

    IDEState *s = opaque;

    uint64_t nb_sectors;



    if (!s->identify_set) {

        return;

    }



    bdrv_get_geometry(s->bs, &nb_sectors);

    s->nb_sectors = nb_sectors;



    /* Update the identify data buffer. */

    if (s->drive_kind == IDE_CFATA) {

        ide_cfata_identify_size(s);

    } else {

        /* IDE_CD uses a different set of callbacks entirely. */

        assert(s->drive_kind != IDE_CD);

        ide_identify_size(s);

    }

}
