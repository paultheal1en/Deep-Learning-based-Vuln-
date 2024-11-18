static void usb_msd_cancel_io(USBPacket *p, void *opaque)

{

    MSDState *s = opaque;

    s->scsi_dev->info->cancel_io(s->scsi_dev, s->tag);

    s->packet = NULL;

    s->scsi_len = 0;

}
