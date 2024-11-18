static void i82801b11_bridge_class_init(ObjectClass *klass, void *data)

{

    PCIDeviceClass *k = PCI_DEVICE_CLASS(klass);

    DeviceClass *dc = DEVICE_CLASS(klass);



    k->is_bridge = 1;

    k->vendor_id = PCI_VENDOR_ID_INTEL;

    k->device_id = PCI_DEVICE_ID_INTEL_82801BA_11;

    k->revision = ICH9_D2P_A2_REVISION;

    k->init = i82801b11_bridge_initfn;

    k->config_write = pci_bridge_write_config;


    set_bit(DEVICE_CATEGORY_BRIDGE, dc->categories);

}