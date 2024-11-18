static PCIDevice *do_pci_register_device(PCIDevice *pci_dev, PCIBus *bus,

                                         const char *name, int devfn,

                                         PCIConfigReadFunc *config_read,

                                         PCIConfigWriteFunc *config_write,

                                         uint8_t header_type)

{

    if (devfn < 0) {

        for(devfn = bus->devfn_min ; devfn < 256; devfn += 8) {

            if (!bus->devices[devfn])

                goto found;

        }

        return NULL;

    found: ;

    } else if (bus->devices[devfn]) {

        return NULL;

    }

    pci_dev->bus = bus;

    pci_dev->devfn = devfn;

    pstrcpy(pci_dev->name, sizeof(pci_dev->name), name);

    memset(pci_dev->irq_state, 0, sizeof(pci_dev->irq_state));

    pci_config_alloc(pci_dev);



    header_type &= ~PCI_HEADER_TYPE_MULTI_FUNCTION;

    if (header_type == PCI_HEADER_TYPE_NORMAL) {

        pci_set_default_subsystem_id(pci_dev);

    }

    pci_init_cmask(pci_dev);

    pci_init_wmask(pci_dev);

    if (header_type == PCI_HEADER_TYPE_BRIDGE) {

        pci_init_wmask_bridge(pci_dev);

    }



    if (!config_read)

        config_read = pci_default_read_config;

    if (!config_write)

        config_write = pci_default_write_config;

    pci_dev->config_read = config_read;

    pci_dev->config_write = config_write;

    bus->devices[devfn] = pci_dev;

    pci_dev->irq = qemu_allocate_irqs(pci_set_irq, pci_dev, PCI_NUM_PINS);

    pci_dev->version_id = 2; /* Current pci device vmstate version */

    return pci_dev;

}
