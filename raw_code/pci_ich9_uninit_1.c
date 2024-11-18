static int pci_ich9_uninit(PCIDevice *dev)

{

    struct AHCIPCIState *d;

    d = DO_UPCAST(struct AHCIPCIState, card, dev);



    if (msi_enabled(dev)) {

        msi_uninit(dev);

    }



    qemu_unregister_reset(ahci_reset, d);

    ahci_uninit(&d->ahci);



    return 0;

}
