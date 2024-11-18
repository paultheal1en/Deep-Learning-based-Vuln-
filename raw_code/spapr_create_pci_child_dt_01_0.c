static int spapr_create_pci_child_dt(sPAPRPHBState *phb, PCIDevice *dev,

                                     void *fdt, int node_offset)

{

    int offset, ret;

    char nodename[FDT_NAME_MAX];



    pci_get_node_name(nodename, FDT_NAME_MAX, dev);

    offset = fdt_add_subnode(fdt, node_offset, nodename);

    ret = spapr_populate_pci_child_dt(dev, fdt, offset, phb);



    g_assert(!ret);

    if (ret) {

        return 0;

    }

    return offset;

}
