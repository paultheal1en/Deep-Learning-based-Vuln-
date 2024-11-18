static int bonito_initfn(PCIDevice *dev)

{

    PCIBonitoState *s = DO_UPCAST(PCIBonitoState, dev, dev);



    /* Bonito North Bridge, built on FPGA, VENDOR_ID/DEVICE_ID are "undefined" */

    pci_config_set_vendor_id(dev->config, 0xdf53);

    pci_config_set_device_id(dev->config, 0x00d5);

    pci_config_set_class(dev->config, PCI_CLASS_BRIDGE_HOST);

    pci_config_set_prog_interface(dev->config, 0x00);

    pci_config_set_revision(dev->config, 0x01);



    /* set the north bridge register mapping */

    s->bonito_reg_handle = cpu_register_io_memory(bonito_read, bonito_write, s,

                                                  DEVICE_NATIVE_ENDIAN);

    s->bonito_reg_start = BONITO_INTERNAL_REG_BASE;

    s->bonito_reg_length = BONITO_INTERNAL_REG_SIZE;

    cpu_register_physical_memory(s->bonito_reg_start, s->bonito_reg_length,

                                 s->bonito_reg_handle);



    /* set the north bridge pci configure  mapping */

    s->bonito_pciconf_handle = cpu_register_io_memory(bonito_pciconf_read,

                                                      bonito_pciconf_write, s,

                                                      DEVICE_NATIVE_ENDIAN);

    s->bonito_pciconf_start = BONITO_PCICONFIG_BASE;

    s->bonito_pciconf_length = BONITO_PCICONFIG_SIZE;

    cpu_register_physical_memory(s->bonito_pciconf_start, s->bonito_pciconf_length,

                                 s->bonito_pciconf_handle);



    /* set the south bridge pci configure  mapping */

    s->bonito_spciconf_handle = cpu_register_io_memory(bonito_spciconf_read,

                                                       bonito_spciconf_write, s,

                                                       DEVICE_NATIVE_ENDIAN);

    s->bonito_spciconf_start = BONITO_SPCICONFIG_BASE;

    s->bonito_spciconf_length = BONITO_SPCICONFIG_SIZE;

    cpu_register_physical_memory(s->bonito_spciconf_start, s->bonito_spciconf_length,

                                 s->bonito_spciconf_handle);



    s->bonito_ldma_handle = cpu_register_io_memory(bonito_ldma_read,

                                                   bonito_ldma_write, s,

                                                   DEVICE_NATIVE_ENDIAN);

    s->bonito_ldma_start = 0xbfe00200;

    s->bonito_ldma_length = 0x100;

    cpu_register_physical_memory(s->bonito_ldma_start, s->bonito_ldma_length,

                                 s->bonito_ldma_handle);



    s->bonito_cop_handle = cpu_register_io_memory(bonito_cop_read,

                                                  bonito_cop_write, s,

                                                  DEVICE_NATIVE_ENDIAN);

    s->bonito_cop_start = 0xbfe00300;

    s->bonito_cop_length = 0x100;

    cpu_register_physical_memory(s->bonito_cop_start, s->bonito_cop_length,

                                 s->bonito_cop_handle);



    /* Map PCI IO Space  0x1fd0 0000 - 0x1fd1 0000 */

    s->bonito_pciio_start = BONITO_PCIIO_BASE;

    s->bonito_pciio_length = BONITO_PCIIO_SIZE;

    isa_mem_base = s->bonito_pciio_start;

    isa_mmio_init(s->bonito_pciio_start, s->bonito_pciio_length);



    /* add pci local io mapping */

    s->bonito_localio_start = BONITO_DEV_BASE;

    s->bonito_localio_length = BONITO_DEV_SIZE;

    isa_mmio_init(s->bonito_localio_start, s->bonito_localio_length);



    /* set the default value of north bridge pci config */

    pci_set_word(dev->config + PCI_COMMAND, 0x0000);

    pci_set_word(dev->config + PCI_STATUS, 0x0000);

    pci_set_word(dev->config + PCI_SUBSYSTEM_VENDOR_ID, 0x0000);

    pci_set_word(dev->config + PCI_SUBSYSTEM_ID, 0x0000);



    pci_set_byte(dev->config + PCI_INTERRUPT_LINE, 0x00);

    pci_set_byte(dev->config + PCI_INTERRUPT_PIN, 0x01);

    pci_set_byte(dev->config + PCI_MIN_GNT, 0x3c);

    pci_set_byte(dev->config + PCI_MAX_LAT, 0x00);



    qemu_register_reset(bonito_reset, s);



    return 0;

}
