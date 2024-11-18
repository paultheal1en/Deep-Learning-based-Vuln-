static void scsi_disk_unit_attention_reported(SCSIDevice *dev)

{

    SCSIDiskState *s = DO_UPCAST(SCSIDiskState, qdev, dev);

    if (s->media_changed) {

        s->media_changed = false;

        s->qdev.unit_attention = SENSE_CODE(MEDIUM_CHANGED);

    }

}
