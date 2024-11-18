static void pci_device_reset(PCIDevice *dev)

{

    int r;



    dev->irq_state = 0;

    pci_update_irq_status(dev);

    /* Clear all writeable bits */

    pci_word_test_and_clear_mask(dev->config + PCI_COMMAND,

                                 pci_get_word(dev->wmask + PCI_COMMAND));

    dev->config[PCI_CACHE_LINE_SIZE] = 0x0;

    dev->config[PCI_INTERRUPT_LINE] = 0x0;

    for (r = 0; r < PCI_NUM_REGIONS; ++r) {

        PCIIORegion *region = &dev->io_regions[r];

        if (!region->size) {

            continue;

        }



        if (!(region->type & PCI_BASE_ADDRESS_SPACE_IO) &&

            region->type & PCI_BASE_ADDRESS_MEM_TYPE_64) {

            pci_set_quad(dev->config + pci_bar(dev, r), region->type);

        } else {

            pci_set_long(dev->config + pci_bar(dev, r), region->type);

        }

    }

    pci_update_mappings(dev);

}
