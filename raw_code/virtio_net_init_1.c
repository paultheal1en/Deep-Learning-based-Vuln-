VirtIODevice *virtio_net_init(DeviceState *dev, NICConf *conf,

                              virtio_net_conf *net)

{

    VirtIONet *n;



    n = (VirtIONet *)virtio_common_init("virtio-net", VIRTIO_ID_NET,

                                        sizeof(struct virtio_net_config),

                                        sizeof(VirtIONet));



    n->vdev.get_config = virtio_net_get_config;

    n->vdev.set_config = virtio_net_set_config;

    n->vdev.get_features = virtio_net_get_features;

    n->vdev.set_features = virtio_net_set_features;

    n->vdev.bad_features = virtio_net_bad_features;

    n->vdev.reset = virtio_net_reset;

    n->vdev.set_status = virtio_net_set_status;

    n->rx_vq = virtio_add_queue(&n->vdev, 256, virtio_net_handle_rx);

    n->tx_vq = virtio_add_queue(&n->vdev, 256, virtio_net_handle_tx);

    n->ctrl_vq = virtio_add_queue(&n->vdev, 64, virtio_net_handle_ctrl);

    qemu_macaddr_default_if_unset(&conf->macaddr);

    memcpy(&n->mac[0], &conf->macaddr, sizeof(n->mac));

    n->status = VIRTIO_NET_S_LINK_UP;



    n->nic = qemu_new_nic(&net_virtio_info, conf, dev->info->name, dev->id, n);



    qemu_format_nic_info_str(&n->nic->nc, conf->macaddr.a);



    n->tx_timer = qemu_new_timer(vm_clock, virtio_net_tx_timer, n);

    n->tx_waiting = 0;

    n->tx_timeout = net->txtimer;

    n->tx_burst = net->txburst;

    n->mergeable_rx_bufs = 0;

    n->promisc = 1; /* for compatibility */



    n->mac_table.macs = qemu_mallocz(MAC_TABLE_ENTRIES * ETH_ALEN);



    n->vlans = qemu_mallocz(MAX_VLAN >> 3);



    n->qdev = dev;

    register_savevm(dev, "virtio-net", -1, VIRTIO_NET_VM_VERSION,

                    virtio_net_save, virtio_net_load, n);

    n->vmstate = qemu_add_vm_change_state_handler(virtio_net_vmstate_change, n);



    return &n->vdev;

}
