static int usb_xhci_initfn(struct PCIDevice *dev)

{

    int i, ret;



    XHCIState *xhci = DO_UPCAST(XHCIState, pci_dev, dev);



    xhci->pci_dev.config[PCI_CLASS_PROG] = 0x30;    /* xHCI */

    xhci->pci_dev.config[PCI_INTERRUPT_PIN] = 0x01; /* interrupt pin 1 */

    xhci->pci_dev.config[PCI_CACHE_LINE_SIZE] = 0x10;

    xhci->pci_dev.config[0x60] = 0x30; /* release number */



    usb_xhci_init(xhci, &dev->qdev);



    if (xhci->numintrs > MAXINTRS) {

        xhci->numintrs = MAXINTRS;





    if (xhci->numintrs < 1) {

        xhci->numintrs = 1;


    if (xhci->numslots > MAXSLOTS) {

        xhci->numslots = MAXSLOTS;


    if (xhci->numslots < 1) {

        xhci->numslots = 1;




    xhci->mfwrap_timer = qemu_new_timer_ns(vm_clock, xhci_mfwrap_timer, xhci);



    xhci->irq = xhci->pci_dev.irq[0];



    memory_region_init(&xhci->mem, "xhci", LEN_REGS);

    memory_region_init_io(&xhci->mem_cap, &xhci_cap_ops, xhci,

                          "capabilities", LEN_CAP);

    memory_region_init_io(&xhci->mem_oper, &xhci_oper_ops, xhci,

                          "operational", 0x400);

    memory_region_init_io(&xhci->mem_runtime, &xhci_runtime_ops, xhci,

                          "runtime", LEN_RUNTIME);

    memory_region_init_io(&xhci->mem_doorbell, &xhci_doorbell_ops, xhci,

                          "doorbell", LEN_DOORBELL);



    memory_region_add_subregion(&xhci->mem, 0,            &xhci->mem_cap);

    memory_region_add_subregion(&xhci->mem, OFF_OPER,     &xhci->mem_oper);

    memory_region_add_subregion(&xhci->mem, OFF_RUNTIME,  &xhci->mem_runtime);

    memory_region_add_subregion(&xhci->mem, OFF_DOORBELL, &xhci->mem_doorbell);



    for (i = 0; i < xhci->numports; i++) {

        XHCIPort *port = &xhci->ports[i];

        uint32_t offset = OFF_OPER + 0x400 + 0x10 * i;

        port->xhci = xhci;

        memory_region_init_io(&port->mem, &xhci_port_ops, port,

                              port->name, 0x10);

        memory_region_add_subregion(&xhci->mem, offset, &port->mem);




    pci_register_bar(&xhci->pci_dev, 0,

                     PCI_BASE_ADDRESS_SPACE_MEMORY|PCI_BASE_ADDRESS_MEM_TYPE_64,

                     &xhci->mem);



    ret = pcie_endpoint_cap_init(&xhci->pci_dev, 0xa0);

    assert(ret >= 0);



    if (xhci->flags & (1 << XHCI_FLAG_USE_MSI)) {

        msi_init(&xhci->pci_dev, 0x70, xhci->numintrs, true, false);


    if (xhci->flags & (1 << XHCI_FLAG_USE_MSI_X)) {

        msix_init(&xhci->pci_dev, xhci->numintrs,

                  &xhci->mem, 0, OFF_MSIX_TABLE,

                  &xhci->mem, 0, OFF_MSIX_PBA,

                  0x90);




    return 0;
