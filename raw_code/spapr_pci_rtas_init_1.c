void spapr_pci_rtas_init(void)

{

    spapr_rtas_register("read-pci-config", rtas_read_pci_config);

    spapr_rtas_register("write-pci-config", rtas_write_pci_config);

    spapr_rtas_register("ibm,read-pci-config", rtas_ibm_read_pci_config);

    spapr_rtas_register("ibm,write-pci-config", rtas_ibm_write_pci_config);

    if (msi_supported) {

        spapr_rtas_register("ibm,query-interrupt-source-number",

                            rtas_ibm_query_interrupt_source_number);

        spapr_rtas_register("ibm,change-msi", rtas_ibm_change_msi);

    }

}
