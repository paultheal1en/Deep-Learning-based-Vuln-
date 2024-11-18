static int i440fx_initfn(PCIDevice *dev)

{

    PCII440FXState *d = DO_UPCAST(PCII440FXState, dev, dev);



    pci_config_set_vendor_id(d->dev.config, PCI_VENDOR_ID_INTEL);

    pci_config_set_device_id(d->dev.config, PCI_DEVICE_ID_INTEL_82441);

    d->dev.config[0x08] = 0x02; // revision

    pci_config_set_class(d->dev.config, PCI_CLASS_BRIDGE_HOST);



    d->dev.config[I440FX_SMRAM] = 0x02;



    cpu_smm_register(&i440fx_set_smm, d);

    return 0;

}
