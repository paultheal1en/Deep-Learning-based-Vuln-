PCIDevice *virtio_net_init(PCIBus *bus, NICInfo *nd, int devfn)

{

    VirtIONet *n;

    static int virtio_net_id;



    n = (VirtIONet *)virtio_init_pci(bus, "virtio-net", 6900, 0x1000,

                                     0, VIRTIO_ID_NET,

                                     0x02, 0x00, 0x00,

                                     6, sizeof(VirtIONet));

    if (!n)

        return NULL;



    n->vdev.get_config = virtio_net_update_config;

    n->vdev.get_features = virtio_net_get_features;

    n->vdev.set_features = virtio_net_set_features;

    n->rx_vq = virtio_add_queue(&n->vdev, 256, virtio_net_handle_rx);

    n->tx_vq = virtio_add_queue(&n->vdev, 256, virtio_net_handle_tx);

    memcpy(n->mac, nd->macaddr, 6);

    n->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                 virtio_net_receive, virtio_net_can_receive, n);



    qemu_format_nic_info_str(n->vc, n->mac);



    n->tx_timer = qemu_new_timer(vm_clock, virtio_net_tx_timer, n);

    n->tx_timer_active = 0;

    n->mergeable_rx_bufs = 0;



    register_savevm("virtio-net", virtio_net_id++, 2,

                    virtio_net_save, virtio_net_load, n);



    return (PCIDevice *)n;

}
