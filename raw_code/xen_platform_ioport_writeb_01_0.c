static void xen_platform_ioport_writeb(void *opaque, hwaddr addr,

                                       uint64_t val, unsigned int size)

{

    PCIXenPlatformState *s = opaque;

    PCIDevice *pci_dev = PCI_DEVICE(s);



    switch (addr) {

    case 0: /* Platform flags */

        platform_fixed_ioport_writeb(opaque, 0, (uint32_t)val);

        break;

    case 4:

        if (val == 1) {

            /*

             * SUSE unplug for Xenlinux

             * xen-kmp used this since xen-3.0.4, instead the official protocol

             * from xen-3.3+ It did an unconditional "outl(1, (ioaddr + 4));"

             * Pre VMDP 1.7 used 4 and 8 depending on how VMDP was configured.

             * If VMDP was to control both disk and LAN it would use 4.

             * If it controlled just disk or just LAN, it would use 8 below.

             */

            pci_unplug_disks(pci_dev->bus);

            pci_unplug_nics(pci_dev->bus);

        }

        break;

    case 8:

        switch (val) {

        case 1:

            pci_unplug_disks(pci_dev->bus);

            break;

        case 2:

            pci_unplug_nics(pci_dev->bus);

            break;

        default:

            log_writeb(s, (uint32_t)val);

            break;

        }

        break;

    default:

        break;

    }

}
