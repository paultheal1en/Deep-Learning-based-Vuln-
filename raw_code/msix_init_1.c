int msix_init(struct PCIDevice *dev, unsigned short nentries,

              MemoryRegion *table_bar, uint8_t table_bar_nr,

              unsigned table_offset, MemoryRegion *pba_bar,

              uint8_t pba_bar_nr, unsigned pba_offset, uint8_t cap_pos)

{

    int cap;

    unsigned table_size, pba_size;

    uint8_t *config;



    /* Nothing to do if MSI is not supported by interrupt controller */

    if (!msi_nonbroken) {

        return -ENOTSUP;

    }



    if (nentries < 1 || nentries > PCI_MSIX_FLAGS_QSIZE + 1) {

        return -EINVAL;

    }



    table_size = nentries * PCI_MSIX_ENTRY_SIZE;

    pba_size = QEMU_ALIGN_UP(nentries, 64) / 8;



    /* Sanity test: table & pba don't overlap, fit within BARs, min aligned */

    if ((table_bar_nr == pba_bar_nr &&

         ranges_overlap(table_offset, table_size, pba_offset, pba_size)) ||

        table_offset + table_size > memory_region_size(table_bar) ||

        pba_offset + pba_size > memory_region_size(pba_bar) ||

        (table_offset | pba_offset) & PCI_MSIX_FLAGS_BIRMASK) {

        return -EINVAL;

    }



    cap = pci_add_capability(dev, PCI_CAP_ID_MSIX, cap_pos, MSIX_CAP_LENGTH);

    if (cap < 0) {

        return cap;

    }



    dev->msix_cap = cap;

    dev->cap_present |= QEMU_PCI_CAP_MSIX;

    config = dev->config + cap;



    pci_set_word(config + PCI_MSIX_FLAGS, nentries - 1);

    dev->msix_entries_nr = nentries;

    dev->msix_function_masked = true;



    pci_set_long(config + PCI_MSIX_TABLE, table_offset | table_bar_nr);

    pci_set_long(config + PCI_MSIX_PBA, pba_offset | pba_bar_nr);



    /* Make flags bit writable. */

    dev->wmask[cap + MSIX_CONTROL_OFFSET] |= MSIX_ENABLE_MASK |

                                             MSIX_MASKALL_MASK;



    dev->msix_table = g_malloc0(table_size);

    dev->msix_pba = g_malloc0(pba_size);

    dev->msix_entry_used = g_malloc0(nentries * sizeof *dev->msix_entry_used);



    msix_mask_all(dev, nentries);



    memory_region_init_io(&dev->msix_table_mmio, OBJECT(dev), &msix_table_mmio_ops, dev,

                          "msix-table", table_size);

    memory_region_add_subregion(table_bar, table_offset, &dev->msix_table_mmio);

    memory_region_init_io(&dev->msix_pba_mmio, OBJECT(dev), &msix_pba_mmio_ops, dev,

                          "msix-pba", pba_size);

    memory_region_add_subregion(pba_bar, pba_offset, &dev->msix_pba_mmio);



    return 0;

}
