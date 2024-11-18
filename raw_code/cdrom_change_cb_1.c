static void cdrom_change_cb(void *opaque, int reason)

{

    IDEState *s = opaque;

    uint64_t nb_sectors;



    if (!(reason & CHANGE_MEDIA)) {

        return;

    }



    bdrv_get_geometry(s->bs, &nb_sectors);

    s->nb_sectors = nb_sectors;



    s->sense_key = SENSE_UNIT_ATTENTION;

    s->asc = ASC_MEDIUM_MAY_HAVE_CHANGED;

    s->cdrom_changed = 1;

    s->events.new_media = true;

    ide_set_irq(s->bus);

}
