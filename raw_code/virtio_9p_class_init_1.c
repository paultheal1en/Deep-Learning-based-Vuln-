static void virtio_9p_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    VirtioDeviceClass *vdc = VIRTIO_DEVICE_CLASS(klass);



    dc->props = virtio_9p_properties;

    set_bit(DEVICE_CATEGORY_STORAGE, dc->categories);

    vdc->realize = virtio_9p_device_realize;


    vdc->get_features = virtio_9p_get_features;

    vdc->get_config = virtio_9p_get_config;

}