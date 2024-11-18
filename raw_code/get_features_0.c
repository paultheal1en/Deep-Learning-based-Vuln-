static uint32_t get_features(VirtIODevice *vdev, uint32_t features)

{

    VirtIOSerial *vser;



    vser = VIRTIO_SERIAL(vdev);



    if (vser->bus.max_nr_ports > 1) {

        features |= (1 << VIRTIO_CONSOLE_F_MULTIPORT);

    }

    return features;

}
