static void assigned_dev_msix_mmio_write(void *opaque, target_phys_addr_t addr,

                                         uint64_t val, unsigned size)

{

    AssignedDevice *adev = opaque;

    PCIDevice *pdev = &adev->dev;

    uint16_t ctrl;

    MSIXTableEntry orig;

    int i = addr >> 4;



    if (i >= adev->msix_max) {

        return; /* Drop write */

    }



    ctrl = pci_get_word(pdev->config + pdev->msix_cap + PCI_MSIX_FLAGS);



    DEBUG("write to MSI-X table offset 0x%lx, val 0x%lx\n", addr, val);



    if (ctrl & PCI_MSIX_FLAGS_ENABLE) {

        orig = adev->msix_table[i];

    }



    memcpy((uint8_t *)adev->msix_table + addr, &val, size);



    if (ctrl & PCI_MSIX_FLAGS_ENABLE) {

        MSIXTableEntry *entry = &adev->msix_table[i];



        if (!assigned_dev_msix_masked(&orig) &&

            assigned_dev_msix_masked(entry)) {

            /*

             * Vector masked, disable it

             *

             * XXX It's not clear if we can or should actually attempt

             * to mask or disable the interrupt.  KVM doesn't have

             * support for pending bits and kvm_assign_set_msix_entry

             * doesn't modify the device hardware mask.  Interrupts

             * while masked are simply not injected to the guest, so

             * are lost.  Can we get away with always injecting an

             * interrupt on unmask?

             */

        } else if (assigned_dev_msix_masked(&orig) &&

                   !assigned_dev_msix_masked(entry)) {

            /* Vector unmasked */

            if (i >= adev->msi_virq_nr || adev->msi_virq[i] < 0) {

                /* Previously unassigned vector, start from scratch */

                assigned_dev_update_msix(pdev);

                return;

            } else {

                /* Update an existing, previously masked vector */

                MSIMessage msg;

                int ret;



                msg.address = entry->addr_lo |

                    ((uint64_t)entry->addr_hi << 32);

                msg.data = entry->data;



                ret = kvm_irqchip_update_msi_route(kvm_state,

                                                   adev->msi_virq[i], msg);

                if (ret) {

                    error_report("Error updating irq routing entry (%d)", ret);

                }

            }

        }

    }

}
