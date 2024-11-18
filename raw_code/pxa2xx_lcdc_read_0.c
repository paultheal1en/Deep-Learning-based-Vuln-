static uint64_t pxa2xx_lcdc_read(void *opaque, hwaddr offset,

                                 unsigned size)

{

    PXA2xxLCDState *s = (PXA2xxLCDState *) opaque;

    int ch;



    switch (offset) {

    case LCCR0:

        return s->control[0];

    case LCCR1:

        return s->control[1];

    case LCCR2:

        return s->control[2];

    case LCCR3:

        return s->control[3];

    case LCCR4:

        return s->control[4];

    case LCCR5:

        return s->control[5];



    case OVL1C1:

        return s->ovl1c[0];

    case OVL1C2:

        return s->ovl1c[1];

    case OVL2C1:

        return s->ovl2c[0];

    case OVL2C2:

        return s->ovl2c[1];



    case CCR:

        return s->ccr;



    case CMDCR:

        return s->cmdcr;



    case TRGBR:

        return s->trgbr;

    case TCR:

        return s->tcr;



    case 0x200 ... 0x1000:	/* DMA per-channel registers */

        ch = (offset - 0x200) >> 4;

        if (!(ch >= 0 && ch < PXA_LCDDMA_CHANS))

            goto fail;



        switch (offset & 0xf) {

        case DMA_FDADR:

            return s->dma_ch[ch].descriptor;

        case DMA_FSADR:

            return s->dma_ch[ch].source;

        case DMA_FIDR:

            return s->dma_ch[ch].id;

        case DMA_LDCMD:

            return s->dma_ch[ch].command;

        default:

            goto fail;

        }



    case FBR0:

        return s->dma_ch[0].branch;

    case FBR1:

        return s->dma_ch[1].branch;

    case FBR2:

        return s->dma_ch[2].branch;

    case FBR3:

        return s->dma_ch[3].branch;

    case FBR4:

        return s->dma_ch[4].branch;

    case FBR5:

        return s->dma_ch[5].branch;

    case FBR6:

        return s->dma_ch[6].branch;



    case BSCNTR:

        return s->bscntr;



    case PRSR:

        return 0;



    case LCSR0:

        return s->status[0];

    case LCSR1:

        return s->status[1];

    case LIIDR:

        return s->liidr;



    default:

    fail:

        hw_error("%s: Bad offset " REG_FMT "\n", __FUNCTION__, offset);

    }



    return 0;

}
