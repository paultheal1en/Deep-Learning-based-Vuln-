static void bcm2835_property_mbox_push(BCM2835PropertyState *s, uint32_t value)

{

    uint32_t tag;

    uint32_t bufsize;

    uint32_t tot_len;

    size_t resplen;

    uint32_t tmp;



    value &= ~0xf;



    s->addr = value;



    tot_len = ldl_phys(&s->dma_as, value);



    /* @(addr + 4) : Buffer response code */

    value = s->addr + 8;

    while (value + 8 <= s->addr + tot_len) {

        tag = ldl_phys(&s->dma_as, value);

        bufsize = ldl_phys(&s->dma_as, value + 4);

        /* @(value + 8) : Request/response indicator */

        resplen = 0;

        switch (tag) {

        case 0x00000000: /* End tag */

            break;

        case 0x00000001: /* Get firmware revision */

            stl_phys(&s->dma_as, value + 12, 346337);

            resplen = 4;

            break;

        case 0x00010001: /* Get board model */

            qemu_log_mask(LOG_UNIMP,

                          "bcm2835_property: %x get board model NYI\n", tag);

            resplen = 4;

            break;

        case 0x00010002: /* Get board revision */

            qemu_log_mask(LOG_UNIMP,

                          "bcm2835_property: %x get board revision NYI\n", tag);

            resplen = 4;

            break;

        case 0x00010003: /* Get board MAC address */

            resplen = sizeof(s->macaddr.a);

            dma_memory_write(&s->dma_as, value + 12, s->macaddr.a, resplen);

            break;

        case 0x00010004: /* Get board serial */

            qemu_log_mask(LOG_UNIMP,

                          "bcm2835_property: %x get board serial NYI\n", tag);

            resplen = 8;

            break;

        case 0x00010005: /* Get ARM memory */

            /* base */

            stl_phys(&s->dma_as, value + 12, 0);

            /* size */

            stl_phys(&s->dma_as, value + 16, s->ram_size);

            resplen = 8;

            break;

        case 0x00028001: /* Set power state */

            /* Assume that whatever device they asked for exists,

             * and we'll just claim we set it to the desired state

             */

            tmp = ldl_phys(&s->dma_as, value + 16);

            stl_phys(&s->dma_as, value + 16, (tmp & 1));

            resplen = 8;

            break;



        /* Clocks */



        case 0x00030001: /* Get clock state */

            stl_phys(&s->dma_as, value + 16, 0x1);

            resplen = 8;

            break;



        case 0x00038001: /* Set clock state */

            qemu_log_mask(LOG_UNIMP,

                          "bcm2835_property: %x set clock state NYI\n", tag);

            resplen = 8;

            break;



        case 0x00030002: /* Get clock rate */

        case 0x00030004: /* Get max clock rate */

        case 0x00030007: /* Get min clock rate */

            switch (ldl_phys(&s->dma_as, value + 12)) {

            case 1: /* EMMC */

                stl_phys(&s->dma_as, value + 16, 50000000);

                break;

            case 2: /* UART */

                stl_phys(&s->dma_as, value + 16, 3000000);

                break;

            default:

                stl_phys(&s->dma_as, value + 16, 700000000);

                break;

            }

            resplen = 8;

            break;



        case 0x00038002: /* Set clock rate */

        case 0x00038004: /* Set max clock rate */

        case 0x00038007: /* Set min clock rate */

            qemu_log_mask(LOG_UNIMP,

                          "bcm2835_property: %x set clock rates NYI\n", tag);

            resplen = 8;

            break;



        /* Temperature */



        case 0x00030006: /* Get temperature */

            stl_phys(&s->dma_as, value + 16, 25000);

            resplen = 8;

            break;



        case 0x0003000A: /* Get max temperature */

            stl_phys(&s->dma_as, value + 16, 99000);

            resplen = 8;

            break;





        case 0x00060001: /* Get DMA channels */

            /* channels 2-5 */

            stl_phys(&s->dma_as, value + 12, 0x003C);

            resplen = 4;

            break;



        case 0x00050001: /* Get command line */

            resplen = 0;

            break;



        default:

            qemu_log_mask(LOG_GUEST_ERROR,

                          "bcm2835_property: unhandled tag %08x\n", tag);

            break;

        }



        if (tag == 0) {

            break;

        }



        stl_phys(&s->dma_as, value + 8, (1 << 31) | resplen);

        value += bufsize + 12;

    }



    /* Buffer response code */

    stl_phys(&s->dma_as, s->addr + 4, (1 << 31));

}
