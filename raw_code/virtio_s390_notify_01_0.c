static void virtio_s390_notify(void *opaque, uint16_t vector)

{

    VirtIOS390Device *dev = (VirtIOS390Device*)opaque;

    uint64_t token = s390_virtio_device_vq_token(dev, vector);



    /* XXX kvm dependency! */

    kvm_s390_virtio_irq(s390_cpu_addr2state(0), 0, token);

}
