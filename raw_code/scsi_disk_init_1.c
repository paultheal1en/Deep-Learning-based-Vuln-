SCSIDevice *scsi_disk_init(BlockDriverState *bdrv, int tcq,

                           scsi_completionfn completion, void *opaque)

{

    SCSIDevice *d;

    SCSIDeviceState *s;




    s = (SCSIDeviceState *)qemu_mallocz(sizeof(SCSIDeviceState));

    s->bdrv = bdrv;

    s->tcq = tcq;

    s->completion = completion;

    s->opaque = opaque;

    if (bdrv_get_type_hint(s->bdrv) == BDRV_TYPE_CDROM) {

        s->cluster_size = 4;

    } else {

        s->cluster_size = 1;

    }

    bdrv_get_geometry(s->bdrv, &nb_sectors);

    nb_sectors /= s->cluster_size;

    if (nb_sectors)

        nb_sectors--;

    s->max_lba = nb_sectors;

    strncpy(s->drive_serial_str, drive_get_serial(s->bdrv),

            sizeof(s->drive_serial_str));

    if (strlen(s->drive_serial_str) == 0)

        pstrcpy(s->drive_serial_str, sizeof(s->drive_serial_str), "0");

    qemu_add_vm_change_state_handler(scsi_dma_restart_cb, s);

    d = (SCSIDevice *)qemu_mallocz(sizeof(SCSIDevice));

    d->state = s;

    d->destroy = scsi_destroy;

    d->send_command = scsi_send_command;

    d->read_data = scsi_read_data;

    d->write_data = scsi_write_data;

    d->cancel_io = scsi_cancel_io;

    d->get_buf = scsi_get_buf;



    return d;

}