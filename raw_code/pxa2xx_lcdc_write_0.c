static void pxa2xx_lcdc_write(void *opaque, hwaddr offset,

                              uint64_t value, unsigned size)

{

    PXA2xxLCDState *s = (PXA2xxLCDState *) opaque;

    int ch;



    switch (offset) {

    case LCCR0:

        /* ACK Quick Disable done */

        if ((s->control[0] & LCCR0_ENB) && !(value & LCCR0_ENB))

            s->status[0] |= LCSR0_QD;



        if (!(s->control[0] & LCCR0_LCDT) && (value & LCCR0_LCDT))

            printf("%s: internal frame buffer unsupported\n", __FUNCTION__);



        if ((s->control[3] & LCCR3_API) &&

                (value & LCCR0_ENB) && !(value & LCCR0_LCDT))

            s->status[0] |= LCSR0_ABC;



        s->control[0] = value & 0x07ffffff;

        pxa2xx_lcdc_int_update(s);



        s->dma_ch[0].up = !!(value & LCCR0_ENB);

        s->dma_ch[1].up = (s->ovl1c[0] & OVLC1_EN) || (value & LCCR0_SDS);

        break;



    case LCCR1:

        s->control[1] = value;

        break;



    case LCCR2:

        s->control[2] = value;

        break;



    case LCCR3:

        s->control[3] = value & 0xefffffff;

        s->bpp = LCCR3_BPP(value);

        break;



    case LCCR4:

        s->control[4] = value & 0x83ff81ff;

        break;



    case LCCR5:

        s->control[5] = value & 0x3f3f3f3f;

        break;



    case OVL1C1:

        if (!(s->ovl1c[0] & OVLC1_EN) && (value & OVLC1_EN))

            printf("%s: Overlay 1 not supported\n", __FUNCTION__);



        s->ovl1c[0] = value & 0x80ffffff;

        s->dma_ch[1].up = (value & OVLC1_EN) || (s->control[0] & LCCR0_SDS);

        break;



    case OVL1C2:

        s->ovl1c[1] = value & 0x000fffff;

        break;



    case OVL2C1:

        if (!(s->ovl2c[0] & OVLC1_EN) && (value & OVLC1_EN))

            printf("%s: Overlay 2 not supported\n", __FUNCTION__);



        s->ovl2c[0] = value & 0x80ffffff;

        s->dma_ch[2].up = !!(value & OVLC1_EN);

        s->dma_ch[3].up = !!(value & OVLC1_EN);

        s->dma_ch[4].up = !!(value & OVLC1_EN);

        break;



    case OVL2C2:

        s->ovl2c[1] = value & 0x007fffff;

        break;



    case CCR:

        if (!(s->ccr & CCR_CEN) && (value & CCR_CEN))

            printf("%s: Hardware cursor unimplemented\n", __FUNCTION__);



        s->ccr = value & 0x81ffffe7;

        s->dma_ch[5].up = !!(value & CCR_CEN);

        break;



    case CMDCR:

        s->cmdcr = value & 0xff;

        break;



    case TRGBR:

        s->trgbr = value & 0x00ffffff;

        break;



    case TCR:

        s->tcr = value & 0x7fff;

        break;



    case 0x200 ... 0x1000:	/* DMA per-channel registers */

        ch = (offset - 0x200) >> 4;

        if (!(ch >= 0 && ch < PXA_LCDDMA_CHANS))

            goto fail;



        switch (offset & 0xf) {

        case DMA_FDADR:

            s->dma_ch[ch].descriptor = value & 0xfffffff0;

            break;



        default:

            goto fail;

        }

        break;



    case FBR0:

        s->dma_ch[0].branch = value & 0xfffffff3;

        break;

    case FBR1:

        s->dma_ch[1].branch = value & 0xfffffff3;

        break;

    case FBR2:

        s->dma_ch[2].branch = value & 0xfffffff3;

        break;

    case FBR3:

        s->dma_ch[3].branch = value & 0xfffffff3;

        break;

    case FBR4:

        s->dma_ch[4].branch = value & 0xfffffff3;

        break;

    case FBR5:

        s->dma_ch[5].branch = value & 0xfffffff3;

        break;

    case FBR6:

        s->dma_ch[6].branch = value & 0xfffffff3;

        break;



    case BSCNTR:

        s->bscntr = value & 0xf;

        break;



    case PRSR:

        break;



    case LCSR0:

        s->status[0] &= ~(value & 0xfff);

        if (value & LCSR0_BER)

            s->status[0] &= ~LCSR0_BERCH(7);

        break;



    case LCSR1:

        s->status[1] &= ~(value & 0x3e3f3f);

        break;



    default:

    fail:

        hw_error("%s: Bad offset " REG_FMT "\n", __FUNCTION__, offset);

    }

}
