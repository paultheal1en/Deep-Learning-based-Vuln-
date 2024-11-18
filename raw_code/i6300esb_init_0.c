static int i6300esb_init(PCIDevice *dev)

{

    I6300State *d = DO_UPCAST(I6300State, dev, dev);

    uint8_t *pci_conf;

    int io_mem;

    static CPUReadMemoryFunc * const mem_read[3] = {

        i6300esb_mem_readb,

        i6300esb_mem_readw,

        i6300esb_mem_readl,

    };

    static CPUWriteMemoryFunc * const mem_write[3] = {

        i6300esb_mem_writeb,

        i6300esb_mem_writew,

        i6300esb_mem_writel,

    };



    i6300esb_debug("I6300State = %p\n", d);



    d->timer = qemu_new_timer_ns(vm_clock, i6300esb_timer_expired, d);

    d->previous_reboot_flag = 0;



    pci_conf = d->dev.config;

    pci_config_set_vendor_id(pci_conf, PCI_VENDOR_ID_INTEL);

    pci_config_set_device_id(pci_conf, PCI_DEVICE_ID_INTEL_ESB_9);

    pci_config_set_class(pci_conf, PCI_CLASS_SYSTEM_OTHER);



    io_mem = cpu_register_io_memory(mem_read, mem_write, d,

                                    DEVICE_NATIVE_ENDIAN);

    pci_register_bar_simple(&d->dev, 0, 0x10, 0, io_mem);

    /* qemu_register_coalesced_mmio (addr, 0x10); ? */



    return 0;

}
