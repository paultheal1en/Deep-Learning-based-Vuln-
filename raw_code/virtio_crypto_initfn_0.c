static void virtio_crypto_initfn(Object *obj)

{

    VirtIOCryptoPCI *dev = VIRTIO_CRYPTO_PCI(obj);



    virtio_instance_init_common(obj, &dev->vdev, sizeof(dev->vdev),

                                TYPE_VIRTIO_CRYPTO);

    object_property_add_alias(obj, "cryptodev", OBJECT(&dev->vdev),

                              "cryptodev", &error_abort);

}
