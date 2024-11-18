type_init(assign_register_types)



static void assigned_dev_load_option_rom(AssignedDevice *dev)

{

    int size = 0;



    pci_assign_dev_load_option_rom(&dev->dev, OBJECT(dev), &size,

                                   dev->host.domain, dev->host.bus,

                                   dev->host.slot, dev->host.function);



    if (!size) {

        error_report("pci-assign: Invalid ROM.");

    }

}
