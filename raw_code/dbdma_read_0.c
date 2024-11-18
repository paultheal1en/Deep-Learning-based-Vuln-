static uint64_t dbdma_read(void *opaque, target_phys_addr_t addr,

                           unsigned size)

{

    uint32_t value;

    int channel = addr >> DBDMA_CHANNEL_SHIFT;

    DBDMAState *s = opaque;

    DBDMA_channel *ch = &s->channels[channel];

    int reg = (addr - (channel << DBDMA_CHANNEL_SHIFT)) >> 2;



    value = ch->regs[reg];



    DBDMA_DPRINTF("readl 0x" TARGET_FMT_plx " => 0x%08x\n", addr, value);

    DBDMA_DPRINTF("channel 0x%x reg 0x%x\n",

                  (uint32_t)addr >> DBDMA_CHANNEL_SHIFT, reg);



    switch(reg) {

    case DBDMA_CONTROL:

        value = 0;

        break;

    case DBDMA_STATUS:

    case DBDMA_CMDPTR_LO:

    case DBDMA_INTR_SEL:

    case DBDMA_BRANCH_SEL:

    case DBDMA_WAIT_SEL:

        /* nothing to do */

        break;

    case DBDMA_XFER_MODE:

    case DBDMA_CMDPTR_HI:

    case DBDMA_DATA2PTR_HI:

    case DBDMA_DATA2PTR_LO:

    case DBDMA_ADDRESS_HI:

    case DBDMA_BRANCH_ADDR_HI:

        /* unused */

        value = 0;

        break;

    case DBDMA_RES1:

    case DBDMA_RES2:

    case DBDMA_RES3:

    case DBDMA_RES4:

        /* reserved */

        break;

    }



    return value;

}
