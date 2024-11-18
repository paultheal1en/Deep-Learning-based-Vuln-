static void virtio_scsi_handle_event(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIOSCSI *s = VIRTIO_SCSI(vdev);



    if (s->ctx && !s->dataplane_started) {

        virtio_scsi_dataplane_start(s);

        return;

    }

    if (s->events_dropped) {

        virtio_scsi_push_event(s, NULL, VIRTIO_SCSI_T_NO_EVENT, 0);

    }

}
