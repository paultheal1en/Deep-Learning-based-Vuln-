static void sdhci_class_init(ObjectClass *klass, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(klass);

    SDHCIClass *k = SDHCI_CLASS(klass);



    dc->vmsd = &sdhci_vmstate;

    dc->props = sdhci_properties;

    dc->reset = sdhci_generic_reset;

    dc->realize = sdhci_realize;



    k->reset = sdhci_reset;

    k->mem_read = sdhci_read;

    k->mem_write = sdhci_write;

    k->send_command = sdhci_send_command;

    k->can_issue_command = sdhci_can_issue_command;

    k->data_transfer = sdhci_data_transfer;

    k->end_data_transfer = sdhci_end_transfer;

    k->do_sdma_single = sdhci_sdma_transfer_single_block;

    k->do_sdma_multi = sdhci_sdma_transfer_multi_blocks;

    k->do_adma = sdhci_do_adma;

    k->read_block_from_card = sdhci_read_block_from_card;

    k->write_block_to_card = sdhci_write_block_to_card;

    k->bdata_read = sdhci_read_dataport;

    k->bdata_write = sdhci_write_dataport;

}
