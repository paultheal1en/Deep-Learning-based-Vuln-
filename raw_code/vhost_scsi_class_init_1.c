static void vhost_scsi_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);

    dc->exit = vhost_scsi_exit;

    dc->props = vhost_scsi_properties;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);

    vdc->init = vhost_scsi_init;

    vdc->get_features = vhost_scsi_get_features;

    vdc->set_config = vhost_scsi_set_config;

    vdc->set_status = vhost_scsi_set_status;

}
