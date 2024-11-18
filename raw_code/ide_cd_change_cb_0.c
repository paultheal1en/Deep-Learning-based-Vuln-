static void ide_cd_change_cb(void *opaque, bool load)

{

    IDEState *s = opaque;

    uint64_t nb_sectors;



    s->tray_open = !load;

    bdrv_get_geometry(s->bs, &nb_sectors);

    s->nb_sectors = nb_sectors;



    /*

     * First indicate to the guest that a CD has been removed.  That's

     * done on the next command the guest sends us.

     *

     * Then we set UNIT_ATTENTION, by which the guest will

     * detect a new CD in the drive.  See ide_atapi_cmd() for details.

     */

    s->cdrom_changed = 1;

    s->events.new_media = true;

    s->events.eject_request = false;

    ide_set_irq(s->bus);

}
