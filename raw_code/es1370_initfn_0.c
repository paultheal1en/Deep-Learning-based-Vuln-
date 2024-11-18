static int es1370_initfn (PCIDevice *dev)

{

    ES1370State *s = DO_UPCAST (ES1370State, dev, dev);

    uint8_t *c = s->dev.config;



    pci_config_set_vendor_id (c, PCI_VENDOR_ID_ENSONIQ);

    pci_config_set_device_id (c, PCI_DEVICE_ID_ENSONIQ_ES1370);

    c[PCI_STATUS + 1] = PCI_STATUS_DEVSEL_SLOW >> 8;

    pci_config_set_class (c, PCI_CLASS_MULTIMEDIA_AUDIO);



#if 1

    c[PCI_SUBSYSTEM_VENDOR_ID] = 0x42;

    c[PCI_SUBSYSTEM_VENDOR_ID + 1] = 0x49;

    c[PCI_SUBSYSTEM_ID] = 0x4c;

    c[PCI_SUBSYSTEM_ID + 1] = 0x4c;

#else

    c[PCI_SUBSYSTEM_VENDOR_ID] = 0x74;

    c[PCI_SUBSYSTEM_VENDOR_ID + 1] = 0x12;

    c[PCI_SUBSYSTEM_ID] = 0x71;

    c[PCI_SUBSYSTEM_ID + 1] = 0x13;

    c[PCI_CAPABILITY_LIST] = 0xdc;

    c[PCI_INTERRUPT_LINE] = 10;

    c[0xdc] = 0x00;

#endif



    /* TODO: RST# value should be 0. */

    c[PCI_INTERRUPT_PIN] = 1;

    c[PCI_MIN_GNT] = 0x0c;

    c[PCI_MAX_LAT] = 0x80;



    pci_register_bar (&s->dev, 0, 256, PCI_BASE_ADDRESS_SPACE_IO, es1370_map);

    qemu_register_reset (es1370_on_reset, s);



    AUD_register_card ("es1370", &s->card);

    es1370_reset (s);

    return 0;

}
