static int virtio_scsi_device_exit(DeviceState *qdev)

{

    VirtIOSCSI *s = VIRTIO_SCSI(qdev);

    VirtIOSCSICommon *vs = VIRTIO_SCSI_COMMON(qdev);



    unregister_savevm(qdev, "virtio-scsi", s);

    return virtio_scsi_common_exit(vs);

}
