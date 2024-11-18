static void virtio_net_save(QEMUFile *f, void *opaque)

{

    VirtIONet *n = opaque;



    virtio_save(&n->vdev, f);



    qemu_put_buffer(f, n->mac, ETH_ALEN);

    qemu_put_be32(f, n->tx_timer_active);

    qemu_put_be32(f, n->mergeable_rx_bufs);

    qemu_put_be16(f, n->status);

    qemu_put_byte(f, n->promisc);

    qemu_put_byte(f, n->allmulti);

    qemu_put_be32(f, n->mac_table.in_use);

    qemu_put_buffer(f, n->mac_table.macs, n->mac_table.in_use * ETH_ALEN);

    qemu_put_buffer(f, (uint8_t *)n->vlans, MAX_VLAN >> 3);

    qemu_put_be32(f, 0); /* vnet-hdr placeholder */

    qemu_put_byte(f, n->mac_table.multi_overflow);

    qemu_put_byte(f, n->mac_table.uni_overflow);

    qemu_put_byte(f, n->alluni);

    qemu_put_byte(f, n->nomulti);

    qemu_put_byte(f, n->nouni);

    qemu_put_byte(f, n->nobcast);

}
