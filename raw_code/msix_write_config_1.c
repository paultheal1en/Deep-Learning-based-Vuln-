void msix_write_config(PCIDevice *dev, uint32_t addr,

                       uint32_t val, int len)

{

    unsigned enable_pos = dev->msix_cap + MSIX_CONTROL_OFFSET;

    int vector;

    bool was_masked;



    if (!range_covers_byte(addr, len, enable_pos)) {

        return;

    }



    was_masked = dev->msix_function_masked;

    msix_update_function_masked(dev);



    if (!msix_enabled(dev)) {

        return;

    }



    pci_device_deassert_intx(dev);



    if (dev->msix_function_masked == was_masked) {

        return;

    }



    for (vector = 0; vector < dev->msix_entries_nr; ++vector) {

        msix_handle_mask_update(dev, vector);

    }

}
