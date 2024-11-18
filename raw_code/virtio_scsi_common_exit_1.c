int virtio_scsi_common_exit(VirtIOSCSICommon *vs)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(vs);



    g_free(vs->cmd_vqs);

    virtio_cleanup(vdev);

    return 0;

}
