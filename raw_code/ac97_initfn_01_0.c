static void ac97_initfn (PCIDevice *dev)

{

    PCIAC97LinkState *d = DO_UPCAST (PCIAC97LinkState, dev, dev);

    AC97LinkState *s = &d->ac97;

    uint8_t *c = d->dev.config;



    s->pci_dev = &d->dev;

    pci_config_set_vendor_id (c, PCI_VENDOR_ID_INTEL); /* ro */

    pci_config_set_device_id (c, PCI_DEVICE_ID_INTEL_82801AA_5); /* ro */



    c[0x04] = 0x00;      /* pcicmd pci command rw, ro */

    c[0x05] = 0x00;



    c[0x06] = 0x80;      /* pcists pci status rwc, ro */

    c[0x07] = 0x02;



    c[0x08] = 0x01;      /* rid revision ro */

    c[0x09] = 0x00;      /* pi programming interface ro */

    pci_config_set_class (c, PCI_CLASS_MULTIMEDIA_AUDIO); /* ro */

    c[PCI_HEADER_TYPE] = PCI_HEADER_TYPE_NORMAL; /* headtyp header type ro */



    c[0x10] = 0x01;      /* nabmar native audio mixer base

                            address rw */

    c[0x11] = 0x00;

    c[0x12] = 0x00;

    c[0x13] = 0x00;



    c[0x14] = 0x01;      /* nabmbar native audio bus mastering

                            base address rw */

    c[0x15] = 0x00;

    c[0x16] = 0x00;

    c[0x17] = 0x00;



    c[0x2c] = 0x86;      /* svid subsystem vendor id rwo */

    c[0x2d] = 0x80;



    c[0x2e] = 0x00;      /* sid subsystem id rwo */

    c[0x2f] = 0x00;



    c[0x3c] = 0x00;      /* intr_ln interrupt line rw */

    c[0x3d] = 0x01;      /* intr_pn interrupt pin ro */



    pci_register_bar (&d->dev, 0, 256 * 4, PCI_ADDRESS_SPACE_IO, ac97_map);

    pci_register_bar (&d->dev, 1, 64 * 4, PCI_ADDRESS_SPACE_IO, ac97_map);

    register_savevm ("ac97", 0, 2, ac97_save, ac97_load, s);

    qemu_register_reset (ac97_on_reset, s);

    AUD_register_card ("ac97", &s->card);

    ac97_on_reset (s);

}
