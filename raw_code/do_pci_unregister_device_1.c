static void do_pci_unregister_device(PCIDevice *pci_dev)

{

    pci_dev->bus->devices[pci_dev->devfn] = NULL;

    pci_config_free(pci_dev);



    memory_region_del_subregion(&pci_dev->bus_master_container_region,

                                &pci_dev->bus_master_enable_region);

    address_space_destroy(&pci_dev->bus_master_as);

}
