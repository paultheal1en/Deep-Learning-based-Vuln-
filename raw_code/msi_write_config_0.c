void msi_write_config(PCIDevice *dev, uint32_t addr, uint32_t val, int len)

{

    uint16_t flags = pci_get_word(dev->config + msi_flags_off(dev));

    bool msi64bit = flags & PCI_MSI_FLAGS_64BIT;

    bool msi_per_vector_mask = flags & PCI_MSI_FLAGS_MASKBIT;

    unsigned int nr_vectors;

    uint8_t log_num_vecs;

    uint8_t log_max_vecs;

    unsigned int vector;

    uint32_t pending;



    if (!ranges_overlap(addr, len, dev->msi_cap, msi_cap_sizeof(flags))) {

        return;

    }



#ifdef MSI_DEBUG

    MSI_DEV_PRINTF(dev, "addr 0x%"PRIx32" val 0x%"PRIx32" len %d\n",

                   addr, val, len);

    MSI_DEV_PRINTF(dev, "ctrl: 0x%"PRIx16" address: 0x%"PRIx32,

                   flags,

                   pci_get_long(dev->config + msi_address_lo_off(dev)));

    if (msi64bit) {

        fprintf(stderr, " address-hi: 0x%"PRIx32,

                pci_get_long(dev->config + msi_address_hi_off(dev)));

    }

    fprintf(stderr, " data: 0x%"PRIx16,

            pci_get_word(dev->config + msi_data_off(dev, msi64bit)));

    if (flags & PCI_MSI_FLAGS_MASKBIT) {

        fprintf(stderr, " mask 0x%"PRIx32" pending 0x%"PRIx32,

                pci_get_long(dev->config + msi_mask_off(dev, msi64bit)),

                pci_get_long(dev->config + msi_pending_off(dev, msi64bit)));

    }

    fprintf(stderr, "\n");

#endif



    if (!(flags & PCI_MSI_FLAGS_ENABLE)) {

        return;

    }



    /*

     * Now MSI is enabled, clear INTx# interrupts.

     * the driver is prohibited from writing enable bit to mask

     * a service request. But the guest OS could do this.

     * So we just discard the interrupts as moderate fallback.

     *

     * 6.8.3.3. Enabling Operation

     *   While enabled for MSI or MSI-X operation, a function is prohibited

     *   from using its INTx# pin (if implemented) to request

     *   service (MSI, MSI-X, and INTx# are mutually exclusive).

     */

    pci_device_deassert_intx(dev);



    /*

     * nr_vectors might be set bigger than capable. So clamp it.

     * This is not legal by spec, so we can do anything we like,

     * just don't crash the host

     */

    log_num_vecs =

        (flags & PCI_MSI_FLAGS_QSIZE) >> (ffs(PCI_MSI_FLAGS_QSIZE) - 1);

    log_max_vecs =

        (flags & PCI_MSI_FLAGS_QMASK) >> (ffs(PCI_MSI_FLAGS_QMASK) - 1);

    if (log_num_vecs > log_max_vecs) {

        flags &= ~PCI_MSI_FLAGS_QSIZE;

        flags |= log_max_vecs << (ffs(PCI_MSI_FLAGS_QSIZE) - 1);

        pci_set_word(dev->config + msi_flags_off(dev), flags);

    }



    if (!msi_per_vector_mask) {

        /* if per vector masking isn't supported,

           there is no pending interrupt. */

        return;

    }



    nr_vectors = msi_nr_vectors(flags);



    /* This will discard pending interrupts, if any. */

    pending = pci_get_long(dev->config + msi_pending_off(dev, msi64bit));

    pending &= 0xffffffff >> (PCI_MSI_VECTORS_MAX - nr_vectors);

    pci_set_long(dev->config + msi_pending_off(dev, msi64bit), pending);



    /* deliver pending interrupts which are unmasked */

    for (vector = 0; vector < nr_vectors; ++vector) {

        if (msi_is_masked(dev, vector) || !(pending & (1U << vector))) {

            continue;

        }



        pci_long_test_and_clear_mask(

            dev->config + msi_pending_off(dev, msi64bit), 1U << vector);

        msi_notify(dev, vector);

    }

}
