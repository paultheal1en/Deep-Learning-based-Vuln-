static void virtio_s390_notify(DeviceState *d, uint16_t vector)

{

    VirtIOS390Device *dev = to_virtio_s390_device_fast(d);

    uint64_t token = s390_virtio_device_vq_token(dev, vector);

    S390CPU *cpu = s390_cpu_addr2state(0);



    s390_virtio_irq(cpu, 0, token);

}
