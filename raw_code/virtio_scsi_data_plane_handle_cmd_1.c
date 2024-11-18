static bool virtio_scsi_data_plane_handle_cmd(VirtIODevice *vdev,

                                              VirtQueue *vq)

{

    VirtIOSCSI *s = (VirtIOSCSI *)vdev;



    assert(s->ctx && s->dataplane_started);

    return virtio_scsi_handle_cmd_vq(s, vq);

}
