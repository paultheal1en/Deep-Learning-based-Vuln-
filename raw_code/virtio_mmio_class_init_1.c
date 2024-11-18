static void virtio_mmio_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);



    dc->realize = virtio_mmio_realizefn;

    dc->reset = virtio_mmio_reset;

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);

    dc->props = virtio_mmio_properties;






}