static void dcr_write_pob (void *opaque, int dcrn, uint32_t val)

{

    ppc4xx_pob_t *pob;



    pob = opaque;

    switch (dcrn) {

    case POB0_BEAR:

        /* Read only */

        break;

    case POB0_BESR0:

    case POB0_BESR1:

        /* Write-clear */

        pob->besr[dcrn - POB0_BESR0] &= ~val;

        break;

    }

}
