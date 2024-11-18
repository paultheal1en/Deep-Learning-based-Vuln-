static bool pcie_aer_msg_root_port(PCIDevice *dev, const PCIEAERMsg *msg)

{

    bool msg_sent;

    uint16_t cmd;

    uint8_t *aer_cap;

    uint32_t root_cmd;

    uint32_t root_status;

    bool msi_trigger;



    msg_sent = false;

    cmd = pci_get_word(dev->config + PCI_COMMAND);

    aer_cap = dev->config + dev->exp.aer_cap;

    root_cmd = pci_get_long(aer_cap + PCI_ERR_ROOT_COMMAND);

    root_status = pci_get_long(aer_cap + PCI_ERR_ROOT_STATUS);

    msi_trigger = false;



    if (cmd & PCI_COMMAND_SERR) {

        /* System Error.

         *

         * The way to report System Error is platform specific and

         * it isn't implemented in qemu right now.

         * So just discard the error for now.

         * OS which cares of aer would receive errors via

         * native aer mechanims, so this wouldn't matter.

         */

    }



    /* Errro Message Received: Root Error Status register */

    switch (msg->severity) {

    case PCI_ERR_ROOT_CMD_COR_EN:

        if (root_status & PCI_ERR_ROOT_COR_RCV) {

            root_status |= PCI_ERR_ROOT_MULTI_COR_RCV;

        } else {

            if (root_cmd & PCI_ERR_ROOT_CMD_COR_EN) {

                msi_trigger = true;

            }

            pci_set_word(aer_cap + PCI_ERR_ROOT_COR_SRC, msg->source_id);

        }

        root_status |= PCI_ERR_ROOT_COR_RCV;

        break;

    case PCI_ERR_ROOT_CMD_NONFATAL_EN:

        if (!(root_status & PCI_ERR_ROOT_NONFATAL_RCV) &&

            root_cmd & PCI_ERR_ROOT_CMD_NONFATAL_EN) {

            msi_trigger = true;

        }

        root_status |= PCI_ERR_ROOT_NONFATAL_RCV;

        break;

    case PCI_ERR_ROOT_CMD_FATAL_EN:

        if (!(root_status & PCI_ERR_ROOT_FATAL_RCV) &&

            root_cmd & PCI_ERR_ROOT_CMD_FATAL_EN) {

            msi_trigger = true;

        }

        if (!(root_status & PCI_ERR_ROOT_UNCOR_RCV)) {

            root_status |= PCI_ERR_ROOT_FIRST_FATAL;

        }

        root_status |= PCI_ERR_ROOT_FATAL_RCV;

        break;

    default:

        abort();

        break;

    }

    if (pcie_aer_msg_is_uncor(msg)) {

        if (root_status & PCI_ERR_ROOT_UNCOR_RCV) {

            root_status |= PCI_ERR_ROOT_MULTI_UNCOR_RCV;

        } else {

            pci_set_word(aer_cap + PCI_ERR_ROOT_SRC, msg->source_id);

        }

        root_status |= PCI_ERR_ROOT_UNCOR_RCV;

    }

    pci_set_long(aer_cap + PCI_ERR_ROOT_STATUS, root_status);



    if (root_cmd & msg->severity) {

        /* 6.2.4.1.2 Interrupt Generation */

        if (msix_enabled(dev)) {

            if (msi_trigger) {

                msix_notify(dev, pcie_aer_root_get_vector(dev));

            }

        } else if (msi_enabled(dev)) {

            if (msi_trigger) {

                msi_notify(dev, pcie_aer_root_get_vector(dev));

            }

        } else {

            qemu_set_irq(dev->irq[dev->exp.aer_intx], 1);

        }

        msg_sent = true;

    }

    return msg_sent;

}
