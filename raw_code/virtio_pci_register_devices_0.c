static void virtio_pci_register_devices(void)

{

    type_register_static(&virtio_blk_info);

    type_register_static_alias(&virtio_blk_info, "virtio-blk");

    type_register_static(&virtio_net_info);

    type_register_static_alias(&virtio_net_info, "virtio-net");

    type_register_static(&virtio_serial_info);

    type_register_static_alias(&virtio_serial_info, "virtio-serial");

    type_register_static(&virtio_balloon_info);

    type_register_static_alias(&virtio_balloon_info, "virtio-balloon");

}
