static void dcr_write_sdram (void *opaque, int dcrn, uint32_t val)

{

    ppc4xx_sdram_t *sdram;



    sdram = opaque;

    switch (dcrn) {

    case SDRAM0_CFGADDR:

        sdram->addr = val;

        break;

    case SDRAM0_CFGDATA:

        switch (sdram->addr) {

        case 0x00: /* SDRAM_BESR0 */

            sdram->besr0 &= ~val;

            break;

        case 0x08: /* SDRAM_BESR1 */

            sdram->besr1 &= ~val;

            break;

        case 0x10: /* SDRAM_BEAR */

            sdram->bear = val;

            break;

        case 0x20: /* SDRAM_CFG */

            val &= 0xFFE00000;

            if (!(sdram->cfg & 0x80000000) && (val & 0x80000000)) {

#ifdef DEBUG_SDRAM

                printf("%s: enable SDRAM controller\n", __func__);

#endif

                /* validate all RAM mappings */

                sdram_map_bcr(sdram);

                sdram->status &= ~0x80000000;

            } else if ((sdram->cfg & 0x80000000) && !(val & 0x80000000)) {

#ifdef DEBUG_SDRAM

                printf("%s: disable SDRAM controller\n", __func__);

#endif

                /* invalidate all RAM mappings */

                sdram_unmap_bcr(sdram);

                sdram->status |= 0x80000000;

            }

            if (!(sdram->cfg & 0x40000000) && (val & 0x40000000))

                sdram->status |= 0x40000000;

            else if ((sdram->cfg & 0x40000000) && !(val & 0x40000000))

                sdram->status &= ~0x40000000;

            sdram->cfg = val;

            break;

        case 0x24: /* SDRAM_STATUS */

            /* Read-only register */

            break;

        case 0x30: /* SDRAM_RTR */

            sdram->rtr = val & 0x3FF80000;

            break;

        case 0x34: /* SDRAM_PMIT */

            sdram->pmit = (val & 0xF8000000) | 0x07C00000;

            break;

        case 0x40: /* SDRAM_B0CR */

            sdram_set_bcr(&sdram->bcr[0], val, sdram->cfg & 0x80000000);

            break;

        case 0x44: /* SDRAM_B1CR */

            sdram_set_bcr(&sdram->bcr[1], val, sdram->cfg & 0x80000000);

            break;

        case 0x48: /* SDRAM_B2CR */

            sdram_set_bcr(&sdram->bcr[2], val, sdram->cfg & 0x80000000);

            break;

        case 0x4C: /* SDRAM_B3CR */

            sdram_set_bcr(&sdram->bcr[3], val, sdram->cfg & 0x80000000);

            break;

        case 0x80: /* SDRAM_TR */

            sdram->tr = val & 0x018FC01F;

            break;

        case 0x94: /* SDRAM_ECCCFG */

            sdram->ecccfg = val & 0x00F00000;

            break;

        case 0x98: /* SDRAM_ECCESR */

            val &= 0xFFF0F000;

            if (sdram->eccesr == 0 && val != 0)

                qemu_irq_raise(sdram->irq);

            else if (sdram->eccesr != 0 && val == 0)

                qemu_irq_lower(sdram->irq);

            sdram->eccesr = val;

            break;

        default: /* Error */

            break;

        }

        break;

    }

}
