void s390_pci_sclp_deconfigure(SCCB *sccb)

{

    PciCfgSccb *psccb = (PciCfgSccb *)sccb;

    S390PCIBusDevice *pbdev = s390_pci_find_dev_by_fid(s390_get_phb(),

                                                       be32_to_cpu(psccb->aid));

    uint16_t rc;



    if (be16_to_cpu(sccb->h.length) < 16) {

        rc = SCLP_RC_INSUFFICIENT_SCCB_LENGTH;

        goto out;

    }



    if (!pbdev) {

        DPRINTF("sclp deconfig no dev found\n");

        rc = SCLP_RC_ADAPTER_ID_NOT_RECOGNIZED;

        goto out;

    }



    switch (pbdev->state) {

    case ZPCI_FS_RESERVED:

        rc = SCLP_RC_ADAPTER_IN_RESERVED_STATE;

        break;

    case ZPCI_FS_STANDBY:

        rc = SCLP_RC_NO_ACTION_REQUIRED;

        break;

    default:

        if (pbdev->summary_ind) {

            pci_dereg_irqs(pbdev);

        }

        if (pbdev->iommu->enabled) {

            pci_dereg_ioat(pbdev->iommu);

        }

        pbdev->state = ZPCI_FS_STANDBY;

        rc = SCLP_RC_NORMAL_COMPLETION;



        if (pbdev->release_timer) {

            qdev_unplug(DEVICE(pbdev->pdev), NULL);

        }

    }

out:

    psccb->header.response_code = cpu_to_be16(rc);

}
