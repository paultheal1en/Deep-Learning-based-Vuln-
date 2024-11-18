static void pci_dev_get_w64(PCIBus *b, PCIDevice *dev, void *opaque)

{

    Range *range = opaque;

    PCIDeviceClass *pc = PCI_DEVICE_GET_CLASS(dev);

    uint16_t cmd = pci_get_word(dev->config + PCI_COMMAND);

    int i;



    if (!(cmd & PCI_COMMAND_MEMORY)) {

        return;

    }



    if (pc->is_bridge) {

        pcibus_t base = pci_bridge_get_base(dev, PCI_BASE_ADDRESS_MEM_PREFETCH);

        pcibus_t limit = pci_bridge_get_limit(dev, PCI_BASE_ADDRESS_MEM_PREFETCH);



        base = MAX(base, 0x1ULL << 32);



        if (limit >= base) {

            Range pref_range;

            pref_range.begin = base;

            pref_range.end = limit + 1;

            range_extend(range, &pref_range);

        }

    }

    for (i = 0; i < PCI_NUM_REGIONS; ++i) {

        PCIIORegion *r = &dev->io_regions[i];

        Range region_range;



        if (!r->size ||

            (r->type & PCI_BASE_ADDRESS_SPACE_IO) ||

            !(r->type & PCI_BASE_ADDRESS_MEM_TYPE_64)) {

            continue;

        }

        region_range.begin = pci_bar_address(dev, i, r->type, r->size);

        region_range.end = region_range.begin + r->size;



        if (region_range.begin == PCI_BAR_UNMAPPED) {

            continue;

        }



        region_range.begin = MAX(region_range.begin, 0x1ULL << 32);



        if (region_range.end - 1 >= region_range.begin) {

            range_extend(range, &region_range);

        }

    }

}
