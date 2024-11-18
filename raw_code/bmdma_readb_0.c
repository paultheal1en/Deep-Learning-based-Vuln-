static uint32_t bmdma_readb(void *opaque, uint32_t addr)

{

    BMDMAState *bm = opaque;

    PCIIDEState *pci_dev = pci_from_bm(bm);

    uint32_t val;



    switch(addr & 3) {

    case 0:

        val = bm->cmd;

        break;

    case 1:

        val = pci_dev->dev.config[MRDMODE];

        break;

    case 2:

        val = bm->status;

        break;

    case 3:

        if (bm->unit == 0) {

            val = pci_dev->dev.config[UDIDETCR0];

        } else {

            val = pci_dev->dev.config[UDIDETCR1];

        }

        break;

    default:

        val = 0xff;

        break;

    }

#ifdef DEBUG_IDE

    printf("bmdma: readb 0x%02x : 0x%02x\n", addr, val);

#endif

    return val;

}
