void pci_register_bar(PCIDevice *pci_dev, int region_num,

                            pcibus_t size, int type,

                            PCIMapIORegionFunc *map_func)

{

    PCIIORegion *r;

    uint32_t addr;

    uint64_t wmask;



    if ((unsigned int)region_num >= PCI_NUM_REGIONS)

        return;



    if (size & (size-1)) {

        fprintf(stderr, "ERROR: PCI region size must be pow2 "

                    "type=0x%x, size=0x%"FMT_PCIBUS"\n", type, size);

        exit(1);

    }



    r = &pci_dev->io_regions[region_num];

    r->addr = PCI_BAR_UNMAPPED;

    r->size = size;

    r->filtered_size = size;

    r->type = type;

    r->map_func = map_func;



    wmask = ~(size - 1);

    addr = pci_bar(pci_dev, region_num);

    if (region_num == PCI_ROM_SLOT) {

        /* ROM enable bit is writeable */

        wmask |= PCI_ROM_ADDRESS_ENABLE;

    }

    pci_set_long(pci_dev->config + addr, type);

    if (!(r->type & PCI_BASE_ADDRESS_SPACE_IO) &&

        r->type & PCI_BASE_ADDRESS_MEM_TYPE_64) {

        pci_set_quad(pci_dev->wmask + addr, wmask);

        pci_set_quad(pci_dev->cmask + addr, ~0ULL);

    } else {

        pci_set_long(pci_dev->wmask + addr, wmask & 0xffffffff);

        pci_set_long(pci_dev->cmask + addr, 0xffffffff);

    }

}
