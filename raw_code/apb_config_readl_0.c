static uint64_t apb_config_readl (void *opaque,

                                  target_phys_addr_t addr, unsigned size)

{

    APBState *s = opaque;

    uint32_t val;



    switch (addr & 0xffff) {

    case 0x30 ... 0x4f: /* DMA error registers */

        val = 0;

        /* XXX: not implemented yet */

        break;

    case 0x200 ... 0x20b: /* IOMMU */

        val = s->iommu[(addr & 0xf) >> 2];

        break;

    case 0x20c ... 0x3ff: /* IOMMU flush */

        val = 0;

        break;

    case 0xc00 ... 0xc3f: /* PCI interrupt control */

        if (addr & 4) {

            val = s->pci_irq_map[(addr & 0x3f) >> 3];

        } else {

            val = 0;

        }

        break;

    case 0x1000 ... 0x1080: /* OBIO interrupt control */

        if (addr & 4) {

            val = s->obio_irq_map[(addr & 0xff) >> 3];

        } else {

            val = 0;

        }

        break;

    case 0x2000 ... 0x202f: /* PCI control */

        val = s->pci_control[(addr & 0x3f) >> 2];

        break;

    case 0xf020 ... 0xf027: /* Reset control */

        if (addr & 4) {

            val = s->reset_control;

        } else {

            val = 0;

        }

        break;

    case 0x5000 ... 0x51cf: /* PIO/DMA diagnostics */

    case 0xa400 ... 0xa67f: /* IOMMU diagnostics */

    case 0xa800 ... 0xa80f: /* Interrupt diagnostics */

    case 0xf000 ... 0xf01f: /* FFB config, memory control */

        /* we don't care */

    default:

        val = 0;

        break;

    }

    APB_DPRINTF("%s: addr " TARGET_FMT_lx " -> %x\n", __func__, addr, val);



    return val;

}
