static void vt82c686b_pm_realize(PCIDevice *dev, Error **errp)

{

    VT686PMState *s = DO_UPCAST(VT686PMState, dev, dev);

    uint8_t *pci_conf;



    pci_conf = s->dev.config;

    pci_set_word(pci_conf + PCI_COMMAND, 0);

    pci_set_word(pci_conf + PCI_STATUS, PCI_STATUS_FAST_BACK |

                 PCI_STATUS_DEVSEL_MEDIUM);



    /* 0x48-0x4B is Power Management I/O Base */

    pci_set_long(pci_conf + 0x48, 0x00000001);



    /* SMB ports:0xeee0~0xeeef */

    s->smb_io_base =((s->smb_io_base & 0xfff0) + 0x0);

    pci_conf[0x90] = s->smb_io_base | 1;

    pci_conf[0x91] = s->smb_io_base >> 8;

    pci_conf[0xd2] = 0x90;

    pm_smbus_init(&s->dev.qdev, &s->smb);

    memory_region_add_subregion(get_system_io(), s->smb_io_base, &s->smb.io);



    apm_init(dev, &s->apm, NULL, s);



    memory_region_init(&s->io, OBJECT(dev), "vt82c686-pm", 64);

    memory_region_set_enabled(&s->io, false);

    memory_region_add_subregion(get_system_io(), 0, &s->io);



    acpi_pm_tmr_init(&s->ar, pm_tmr_timer, &s->io);

    acpi_pm1_evt_init(&s->ar, pm_tmr_timer, &s->io);

    acpi_pm1_cnt_init(&s->ar, &s->io, 2);

}
