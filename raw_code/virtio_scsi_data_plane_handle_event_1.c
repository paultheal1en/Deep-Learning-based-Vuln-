static bool virtio_scsi_data_plane_handle_event(VirtIODevice *vdev,

                                                VirtQueue *vq)

{

    VirtIOSCSI *s = VIRTIO_SCSI(vdev);



    assert(s->ctx && s->dataplane_started);

    return virtio_scsi_handle_event_vq(s, vq);

}
