static void virtio_scsi_reset(VirtIODevice *vdev)

{

    VirtIOSCSI *s = VIRTIO_SCSI(vdev);

    VirtIOSCSICommon *vs = VIRTIO_SCSI_COMMON(vdev);



    if (s->ctx) {

        virtio_scsi_dataplane_stop(s);

    }

    s->resetting++;

    qbus_reset_all(&s->bus.qbus);

    s->resetting--;



    vs->sense_size = VIRTIO_SCSI_SENSE_DEFAULT_SIZE;

    vs->cdb_size = VIRTIO_SCSI_CDB_DEFAULT_SIZE;

    s->events_dropped = false;

}
