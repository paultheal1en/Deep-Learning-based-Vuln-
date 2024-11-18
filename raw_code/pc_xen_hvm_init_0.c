static void pc_xen_hvm_init(QEMUMachineInitArgs *args)

{

    if (xen_hvm_init() != 0) {

        hw_error("xen hardware virtual machine initialisation failed");

    }

    pc_init_pci(args);

}
