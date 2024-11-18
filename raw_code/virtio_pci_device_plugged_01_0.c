static void virtio_pci_device_plugged(DeviceState *d)

{

    VirtIOPCIProxy *proxy = VIRTIO_PCI(d);

    VirtioBusState *bus = &proxy->bus;

    uint8_t *config;

    uint32_t size;



    config = proxy->pci_dev.config;

    if (proxy->class_code) {

        pci_config_set_class(config, proxy->class_code);

    }

    pci_set_word(config + PCI_SUBSYSTEM_VENDOR_ID,

                 pci_get_word(config + PCI_VENDOR_ID));

    pci_set_word(config + PCI_SUBSYSTEM_ID, virtio_bus_get_vdev_id(bus));

    config[PCI_INTERRUPT_PIN] = 1;



    if (proxy->nvectors &&

        msix_init_exclusive_bar(&proxy->pci_dev, proxy->nvectors, 1)) {

        error_report("unable to init msix vectors to %" PRIu32,

                     proxy->nvectors);

        proxy->nvectors = 0;

    }



    proxy->pci_dev.config_write = virtio_write_config;



    size = VIRTIO_PCI_REGION_SIZE(&proxy->pci_dev)

         + virtio_bus_get_vdev_config_len(bus);

    if (size & (size - 1)) {

        size = 1 << qemu_fls(size);

    }



    memory_region_init_io(&proxy->bar, OBJECT(proxy), &virtio_pci_config_ops,

                          proxy, "virtio-pci", size);

    pci_register_bar(&proxy->pci_dev, 0, PCI_BASE_ADDRESS_SPACE_IO,

                     &proxy->bar);



    if (!kvm_has_many_ioeventfds()) {

        proxy->flags &= ~VIRTIO_PCI_FLAG_USE_IOEVENTFD;

    }



    proxy->host_features |= 0x1 << VIRTIO_F_NOTIFY_ON_EMPTY;

    proxy->host_features |= 0x1 << VIRTIO_F_BAD_FEATURE;

    proxy->host_features = virtio_bus_get_vdev_features(bus,

                                                      proxy->host_features);

}
