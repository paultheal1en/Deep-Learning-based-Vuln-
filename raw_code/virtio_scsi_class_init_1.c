static void virtio_scsi_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);

    dc->exit = virtio_scsi_device_exit;

    dc->props = virtio_scsi_properties;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);

    vdc->init = virtio_scsi_device_init;

    vdc->set_config = virtio_scsi_set_config;

    vdc->get_features = virtio_scsi_get_features;

    vdc->reset = virtio_scsi_reset;

}
