int xen_pt_msix_init(XenPCIPassthroughState *s, uint32_t base)

{

    uint8_t id = 0;

    uint16_t control = 0;

    uint32_t table_off = 0;

    int i, total_entries, bar_index;

    XenHostPCIDevice *hd = &s->real_device;

    PCIDevice *d = &s->dev;

    int fd = -1;

    XenPTMSIX *msix = NULL;

    int rc = 0;



    rc = xen_host_pci_get_byte(hd, base + PCI_CAP_LIST_ID, &id);

    if (rc) {

        return rc;

    }



    if (id != PCI_CAP_ID_MSIX) {

        XEN_PT_ERR(d, "Invalid id %#x base %#x\n", id, base);

        return -1;

    }



    xen_host_pci_get_word(hd, base + PCI_MSIX_FLAGS, &control);

    total_entries = control & PCI_MSIX_FLAGS_QSIZE;

    total_entries += 1;



    s->msix = g_malloc0(sizeof (XenPTMSIX)

                        + total_entries * sizeof (XenPTMSIXEntry));

    msix = s->msix;



    msix->total_entries = total_entries;

    for (i = 0; i < total_entries; i++) {

        msix->msix_entry[i].pirq = XEN_PT_UNASSIGNED_PIRQ;

    }



    memory_region_init_io(&msix->mmio, OBJECT(s), &pci_msix_ops,

                          s, "xen-pci-pt-msix",

                          (total_entries * PCI_MSIX_ENTRY_SIZE

                           + XC_PAGE_SIZE - 1)

                          & XC_PAGE_MASK);



    xen_host_pci_get_long(hd, base + PCI_MSIX_TABLE, &table_off);

    bar_index = msix->bar_index = table_off & PCI_MSIX_FLAGS_BIRMASK;

    table_off = table_off & ~PCI_MSIX_FLAGS_BIRMASK;

    msix->table_base = s->real_device.io_regions[bar_index].base_addr;

    XEN_PT_LOG(d, "get MSI-X table BAR base 0x%"PRIx64"\n", msix->table_base);



    fd = open("/dev/mem", O_RDWR);

    if (fd == -1) {

        rc = -errno;

        XEN_PT_ERR(d, "Can't open /dev/mem: %s\n", strerror(errno));

        goto error_out;

    }

    XEN_PT_LOG(d, "table_off = %#x, total_entries = %d\n",

               table_off, total_entries);

    msix->table_offset_adjust = table_off & 0x0fff;

    msix->phys_iomem_base =

        mmap(NULL,

             total_entries * PCI_MSIX_ENTRY_SIZE + msix->table_offset_adjust,

             PROT_READ,

             MAP_SHARED | MAP_LOCKED,

             fd,

             msix->table_base + table_off - msix->table_offset_adjust);

    close(fd);

    if (msix->phys_iomem_base == MAP_FAILED) {

        rc = -errno;

        XEN_PT_ERR(d, "Can't map physical MSI-X table: %s\n", strerror(errno));

        goto error_out;

    }

    msix->phys_iomem_base = (char *)msix->phys_iomem_base

        + msix->table_offset_adjust;



    XEN_PT_LOG(d, "mapping physical MSI-X table to %p\n",

               msix->phys_iomem_base);



    memory_region_add_subregion_overlap(&s->bar[bar_index], table_off,

                                        &msix->mmio,

                                        2); /* Priority: pci default + 1 */



    return 0;



error_out:

    g_free(s->msix);

    s->msix = NULL;

    return rc;

}
