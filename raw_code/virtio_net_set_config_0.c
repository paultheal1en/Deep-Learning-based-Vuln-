static void virtio_net_set_config(VirtIODevice *vdev, const uint8_t *config)

{

    VirtIONet *n = VIRTIO_NET(vdev);

    struct virtio_net_config netcfg = {};



    memcpy(&netcfg, config, n->config_size);



    if (!(vdev->guest_features >> VIRTIO_NET_F_CTRL_MAC_ADDR & 1) &&

        memcmp(netcfg.mac, n->mac, ETH_ALEN)) {

        memcpy(n->mac, netcfg.mac, ETH_ALEN);

        qemu_format_nic_info_str(qemu_get_queue(n->nic), n->mac);

    }

}
