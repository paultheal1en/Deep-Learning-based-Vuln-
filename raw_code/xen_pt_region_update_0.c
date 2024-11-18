static void xen_pt_region_update(XenPCIPassthroughState *s,

                                 MemoryRegionSection *sec, bool adding)

{

    PCIDevice *d = &s->dev;

    MemoryRegion *mr = sec->mr;

    int bar = -1;

    int rc;

    int op = adding ? DPCI_ADD_MAPPING : DPCI_REMOVE_MAPPING;

    struct CheckBarArgs args = {

        .s = s,

        .addr = sec->offset_within_address_space,

        .size = int128_get64(sec->size),

        .rc = false,

    };



    bar = xen_pt_bar_from_region(s, mr);

    if (bar == -1 && (!s->msix || &s->msix->mmio != mr)) {

        return;

    }



    if (s->msix && &s->msix->mmio == mr) {

        if (adding) {

            s->msix->mmio_base_addr = sec->offset_within_address_space;

            rc = xen_pt_msix_update_remap(s, s->msix->bar_index);

        }

        return;

    }



    args.type = d->io_regions[bar].type;

    pci_for_each_device(d->bus, pci_bus_num(d->bus),

                        xen_pt_check_bar_overlap, &args);

    if (args.rc) {

        XEN_PT_WARN(d, "Region: %d (addr: %#"FMT_PCIBUS

                    ", len: %#"FMT_PCIBUS") is overlapped.\n",

                    bar, sec->offset_within_address_space,

                    int128_get64(sec->size));

    }



    if (d->io_regions[bar].type & PCI_BASE_ADDRESS_SPACE_IO) {

        uint32_t guest_port = sec->offset_within_address_space;

        uint32_t machine_port = s->bases[bar].access.pio_base;

        uint32_t size = int128_get64(sec->size);

        rc = xc_domain_ioport_mapping(xen_xc, xen_domid,

                                      guest_port, machine_port, size,

                                      op);

        if (rc) {

            XEN_PT_ERR(d, "%s ioport mapping failed! (err: %i)\n",

                       adding ? "create new" : "remove old", errno);

        }

    } else {

        pcibus_t guest_addr = sec->offset_within_address_space;

        pcibus_t machine_addr = s->bases[bar].access.maddr

            + sec->offset_within_region;

        pcibus_t size = int128_get64(sec->size);

        rc = xc_domain_memory_mapping(xen_xc, xen_domid,

                                      XEN_PFN(guest_addr + XC_PAGE_SIZE - 1),

                                      XEN_PFN(machine_addr + XC_PAGE_SIZE - 1),

                                      XEN_PFN(size + XC_PAGE_SIZE - 1),

                                      op);

        if (rc) {

            XEN_PT_ERR(d, "%s mem mapping failed! (err: %i)\n",

                       adding ? "create new" : "remove old", errno);

        }

    }

}
