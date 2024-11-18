bool virtio_scsi_handle_event_vq(VirtIOSCSI *s, VirtQueue *vq)

{

    virtio_scsi_acquire(s);

    if (s->events_dropped) {

        virtio_scsi_push_event(s, NULL, VIRTIO_SCSI_T_NO_EVENT, 0);

        virtio_scsi_release(s);

        return true;

    }

    virtio_scsi_release(s);

    return false;

}
