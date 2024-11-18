static int spapr_phb_vfio_eeh_set_option(sPAPRPHBState *sphb,

                                         unsigned int addr, int option)

{

    sPAPRPHBVFIOState *svphb = SPAPR_PCI_VFIO_HOST_BRIDGE(sphb);

    struct vfio_eeh_pe_op op = { .argsz = sizeof(op) };

    int ret;



    switch (option) {

    case RTAS_EEH_DISABLE:

        op.op = VFIO_EEH_PE_DISABLE;

        break;

    case RTAS_EEH_ENABLE: {

        PCIHostState *phb;

        PCIDevice *pdev;



        /*

         * The EEH functionality is enabled on basis of PCI device,

         * instead of PE. We need check the validity of the PCI

         * device address.

         */

        phb = PCI_HOST_BRIDGE(sphb);

        pdev = pci_find_device(phb->bus,

                               (addr >> 16) & 0xFF, (addr >> 8) & 0xFF);

        if (!pdev) {

            return RTAS_OUT_PARAM_ERROR;

        }



        op.op = VFIO_EEH_PE_ENABLE;

        break;

    }

    case RTAS_EEH_THAW_IO:

        op.op = VFIO_EEH_PE_UNFREEZE_IO;

        break;

    case RTAS_EEH_THAW_DMA:

        op.op = VFIO_EEH_PE_UNFREEZE_DMA;

        break;

    default:

        return RTAS_OUT_PARAM_ERROR;

    }



    ret = vfio_container_ioctl(&svphb->phb.iommu_as, svphb->iommugroupid,

                               VFIO_EEH_PE_OP, &op);

    if (ret < 0) {

        return RTAS_OUT_HW_ERROR;

    }



    return RTAS_OUT_SUCCESS;

}
