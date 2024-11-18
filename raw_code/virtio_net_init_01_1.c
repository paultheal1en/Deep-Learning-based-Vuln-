PCIDevice *virtio_net_init(PCIBus *bus, NICInfo *nd, int devfn)

{

    VirtIONet *n;

    static int virtio_net_id;



    n = (VirtIONet *)virtio_init_pci(bus, "virtio-net",

                                     PCI_VENDOR_ID_REDHAT_QUMRANET,

                                     PCI_DEVICE_ID_VIRTIO_NET,

                                     PCI_VENDOR_ID_REDHAT_QUMRANET,

                                     VIRTIO_ID_NET,

                                     PCI_CLASS_NETWORK_ETHERNET, 0x00,

                                     sizeof(struct virtio_net_config),

                                     sizeof(VirtIONet));

    if (!n)

        return NULL;



    n->vdev.get_config = virtio_net_get_config;

    n->vdev.set_config = virtio_net_set_config;

    n->vdev.get_features = virtio_net_get_features;

    n->vdev.set_features = virtio_net_set_features;

    n->vdev.bad_features = virtio_net_bad_features;

    n->vdev.reset = virtio_net_reset;

    n->rx_vq = virtio_add_queue(&n->vdev, 256, virtio_net_handle_rx);

    n->tx_vq = virtio_add_queue(&n->vdev, 256, virtio_net_handle_tx);

    n->ctrl_vq = virtio_add_queue(&n->vdev, 16, virtio_net_handle_ctrl);

    memcpy(n->mac, nd->macaddr, ETH_ALEN);

    n->status = VIRTIO_NET_S_LINK_UP;

    n->vc = qemu_new_vlan_client(nd->vlan, nd->model, nd->name,

                                 virtio_net_receive, virtio_net_can_receive, n);

    n->vc->link_status_changed = virtio_net_set_link_status;



    qemu_format_nic_info_str(n->vc, n->mac);



    n->tx_timer = qemu_new_timer(vm_clock, virtio_net_tx_timer, n);

    n->tx_timer_active = 0;

    n->mergeable_rx_bufs = 0;

    n->promisc = 1; /* for compatibility */



    n->mac_table.macs = qemu_mallocz(MAC_TABLE_ENTRIES * ETH_ALEN);



    n->vlans = qemu_mallocz(MAX_VLAN >> 3);



    register_savevm("virtio-net", virtio_net_id++, VIRTIO_NET_VM_VERSION,

                    virtio_net_save, virtio_net_load, n);

    return (PCIDevice *)n;

}
