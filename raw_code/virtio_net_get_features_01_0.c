static uint32_t virtio_net_get_features(VirtIODevice *vdev)

{

    uint32_t features = (1 << VIRTIO_NET_F_MAC);



    return features;

}
