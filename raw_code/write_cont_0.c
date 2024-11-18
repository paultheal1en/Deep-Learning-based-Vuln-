static void write_cont (void *opaque, uint32_t nport, uint32_t data)

{

    struct dma_cont *d = opaque;

    int iport, ichan;



    iport = (nport >> d->dshift) & 0x0f;

    switch (iport) {

    case 8:                     /* command */

        if (data && (data | CMD_NOT_SUPPORTED)) {

            log ("command %#x not supported\n", data);

            goto error;

        }

        d->command = data;

        break;



    case 9:

        ichan = data & 3;

        if (data & 4) {

            d->status |= 1 << (ichan + 4);

        }

        else {

            d->status &= ~(1 << (ichan + 4));

        }

        d->status &= ~(1 << ichan);

        break;



    case 0xa:                   /* single mask */

        if (data & 4)

            d->mask |= 1 << (data & 3);

        else

            d->mask &= ~(1 << (data & 3));

        break;



    case 0xb:                   /* mode */

        {

            ichan = data & 3;

#ifdef DEBUG_DMA

            int op;

            int ai;

            int dir;

            int opmode;



            op = (data >> 2) & 3;

            ai = (data >> 4) & 1;

            dir = (data >> 5) & 1;

            opmode = (data >> 6) & 3;



            linfo ("ichan %d, op %d, ai %d, dir %d, opmode %d\n",

                   ichan, op, ai, dir, opmode);

#endif



            d->regs[ichan].mode = data;

            break;

        }



    case 0xc:                   /* clear flip flop */

        d->flip_flop = 0;

        break;



    case 0xd:                   /* reset */

        d->flip_flop = 0;

        d->mask = ~0;

        d->status = 0;

        d->command = 0;

        break;



    case 0xe:                   /* clear mask for all channels */

        d->mask = 0;

        break;



    case 0xf:                   /* write mask for all channels */

        d->mask = data;

        break;



    default:

        log ("dma: unknown iport %#x\n", iport);

        goto error;

    }



#ifdef DEBUG_DMA

    if (0xc != iport) {

        linfo ("nport %#06x, ichan % 2d, val %#06x\n",

               nport, ichan, data);

    }

#endif

    return;



 error:

    abort ();

}
