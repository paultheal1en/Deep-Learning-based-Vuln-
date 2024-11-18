static void apb_config_writel (void *opaque, target_phys_addr_t addr,

                               uint64_t val, unsigned size)

{

    APBState *s = opaque;



    APB_DPRINTF("%s: addr " TARGET_FMT_lx " val %" PRIx64 "\n", __func__, addr, val);



    switch (addr & 0xffff) {

    case 0x30 ... 0x4f: /* DMA error registers */

        /* XXX: not implemented yet */

        break;

    case 0x200 ... 0x20b: /* IOMMU */

        s->iommu[(addr & 0xf) >> 2] = val;

        break;

    case 0x20c ... 0x3ff: /* IOMMU flush */

        break;

    case 0xc00 ... 0xc3f: /* PCI interrupt control */

        if (addr & 4) {

            s->pci_irq_map[(addr & 0x3f) >> 3] &= PBM_PCI_IMR_MASK;

            s->pci_irq_map[(addr & 0x3f) >> 3] |= val & ~PBM_PCI_IMR_MASK;

        }

        break;

    case 0x1000 ... 0x1080: /* OBIO interrupt control */

        if (addr & 4) {

            s->obio_irq_map[(addr & 0xff) >> 3] &= PBM_PCI_IMR_MASK;

            s->obio_irq_map[(addr & 0xff) >> 3] |= val & ~PBM_PCI_IMR_MASK;

        }

        break;

    case 0x1400 ... 0x143f: /* PCI interrupt clear */

        if (addr & 4) {

            pci_apb_set_irq(s, (addr & 0x3f) >> 3, 0);

        }

        break;

    case 0x1800 ... 0x1860: /* OBIO interrupt clear */

        if (addr & 4) {

            pci_apb_set_irq(s, 0x20 | ((addr & 0xff) >> 3), 0);

        }

        break;

    case 0x2000 ... 0x202f: /* PCI control */

        s->pci_control[(addr & 0x3f) >> 2] = val;

        break;

    case 0xf020 ... 0xf027: /* Reset control */

        if (addr & 4) {

            val &= RESET_MASK;

            s->reset_control &= ~(val & RESET_WCMASK);

            s->reset_control |= val & RESET_WMASK;

            if (val & SOFT_POR) {

                s->nr_resets = 0;

                qemu_system_reset_request();

            } else if (val & SOFT_XIR) {

                qemu_system_reset_request();

            }

        }

        break;

    case 0x5000 ... 0x51cf: /* PIO/DMA diagnostics */

    case 0xa400 ... 0xa67f: /* IOMMU diagnostics */

    case 0xa800 ... 0xa80f: /* Interrupt diagnostics */

    case 0xf000 ... 0xf01f: /* FFB config, memory control */

        /* we don't care */

    default:

        break;

    }

}
