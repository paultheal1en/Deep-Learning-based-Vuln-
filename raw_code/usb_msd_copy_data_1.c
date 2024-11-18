static void usb_msd_copy_data(MSDState *s)

{

    uint32_t len;

    len = s->usb_len;

    if (len > s->scsi_len)

        len = s->scsi_len;

    if (s->mode == USB_MSDM_DATAIN) {

        memcpy(s->usb_buf, s->scsi_buf, len);

    } else {

        memcpy(s->scsi_buf, s->usb_buf, len);

    }

    s->usb_len -= len;

    s->scsi_len -= len;

    s->usb_buf += len;

    s->scsi_buf += len;

    s->data_len -= len;

    if (s->scsi_len == 0 || s->data_len == 0) {

        if (s->mode == USB_MSDM_DATAIN) {

            s->scsi_dev->info->read_data(s->scsi_dev, s->tag);

        } else if (s->mode == USB_MSDM_DATAOUT) {

            s->scsi_dev->info->write_data(s->scsi_dev, s->tag);

        }

    }

}
