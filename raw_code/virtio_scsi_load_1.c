static int virtio_scsi_load(QEMUFile *f, void *opaque, int version_id)

{

    VirtIOSCSI *s = opaque;

    virtio_load(&s->vdev, f);

    return 0;

}
