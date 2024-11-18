PCIDevice *pci_nic_init_nofail(NICInfo *nd, PCIBus *rootbus,

                               const char *default_model,

                               const char *default_devaddr)

{

    Error *err = NULL;

    PCIDevice *res;



    if (qemu_show_nic_models(nd->model, pci_nic_models))

        exit(0);



    res = pci_nic_init(nd, rootbus, default_model, default_devaddr, &err);

    if (!res) {

        error_report_err(err);

        exit(1);

    }

    return res;

}
