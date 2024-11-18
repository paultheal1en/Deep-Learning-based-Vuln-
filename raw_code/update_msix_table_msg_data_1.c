static void update_msix_table_msg_data(S390PCIBusDevice *pbdev, uint64_t offset,

                                       uint64_t *data, uint8_t len)

{

    uint32_t val;

    uint8_t *msg_data;



    if (offset % PCI_MSIX_ENTRY_SIZE != 8) {

        return;

    }



    if (len != 4) {

        DPRINTF("access msix table msg data but len is %d\n", len);

        return;

    }



    msg_data = (uint8_t *)data - offset % PCI_MSIX_ENTRY_SIZE +

               PCI_MSIX_ENTRY_VECTOR_CTRL;

    val = pci_get_long(msg_data) | (pbdev->fid << ZPCI_MSI_VEC_BITS);

    pci_set_long(msg_data, val);

    DPRINTF("update msix msg_data to 0x%" PRIx64 "\n", *data);

}
