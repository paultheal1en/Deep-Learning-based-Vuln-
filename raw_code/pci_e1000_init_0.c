static int pci_e1000_init(PCIDevice *pci_dev)

{

    E1000State *d = DO_UPCAST(E1000State, dev, pci_dev);

    uint8_t *pci_conf;

    uint16_t checksum = 0;

    int i;

    uint8_t *macaddr;



    pci_conf = d->dev.config;



    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_INTEL);

    pci_config_set_device_id(pci_conf, E1000_DEVID);

    *(uint16_t *)(pci_conf+0x04) = cpu_to_le16(0x0407);

    *(uint16_t *)(pci_conf+0x06) = cpu_to_le16(0x0010);

    pci_conf[0x08] = 0x03;

    pci_config_set_class(pci_conf, PCI_CLASS_NETWORK_ETHERNET);

    pci_conf[0x0c] = 0x10;



    pci_conf[0x3d] = 1; // interrupt pin 0



    d->mmio_index = cpu_register_io_memory(e1000_mmio_read,

            e1000_mmio_write, d);



    pci_register_bar((PCIDevice *)d, 0, PNPMMIO_SIZE,

                           PCI_ADDRESS_SPACE_MEM, e1000_mmio_map);



    pci_register_bar((PCIDevice *)d, 1, IOPORT_SIZE,

                           PCI_ADDRESS_SPACE_IO, ioport_map);



    memmove(d->eeprom_data, e1000_eeprom_template,

        sizeof e1000_eeprom_template);

    qemu_macaddr_default_if_unset(&d->conf.macaddr);

    macaddr = d->conf.macaddr.a;

    for (i = 0; i < 3; i++)

        d->eeprom_data[i] = (macaddr[2*i+1]<<8) | macaddr[2*i];

    for (i = 0; i < EEPROM_CHECKSUM_REG; i++)

        checksum += d->eeprom_data[i];

    checksum = (uint16_t) EEPROM_SUM - checksum;

    d->eeprom_data[EEPROM_CHECKSUM_REG] = checksum;



    d->vc = qemu_new_vlan_client(NET_CLIENT_TYPE_NIC,

                                 d->conf.vlan, d->conf.peer,

                                 d->dev.qdev.info->name, d->dev.qdev.id,

                                 e1000_can_receive, e1000_receive, NULL,

                                 NULL, e1000_cleanup, d);

    d->vc->link_status_changed = e1000_set_link_status;



    qemu_format_nic_info_str(d->vc, macaddr);



    vmstate_register(-1, &vmstate_e1000, d);

    e1000_reset(d);



#if 0 /* rom bev support is broken -> can't load unconditionally */

    if (!pci_dev->qdev.hotplugged) {

        static int loaded = 0;

        if (!loaded) {

            rom_add_option("pxe-e1000.bin");

            loaded = 1;

        }

    }

#endif

    return 0;

}
