static void openpic_reset (void *opaque)

{

    openpic_t *opp = (openpic_t *)opaque;

    int i;



    opp->glbc = 0x80000000;

    /* Initialise controller registers */

    opp->frep = ((OPENPIC_EXT_IRQ - 1) << 16) | ((MAX_CPU - 1) << 8) | VID;

    opp->veni = VENI;

    opp->pint = 0x00000000;

    opp->spve = 0x000000FF;

    opp->tifr = 0x003F7A00;

    /* ? */

    opp->micr = 0x00000000;

    /* Initialise IRQ sources */

    for (i = 0; i < opp->max_irq; i++) {

	opp->src[i].ipvp = 0xA0000000;

	opp->src[i].ide  = 0x00000000;

    }

    /* Initialise IRQ destinations */

    for (i = 0; i < MAX_CPU; i++) {

	opp->dst[i].pctp      = 0x0000000F;

	opp->dst[i].pcsr      = 0x00000000;

	memset(&opp->dst[i].raised, 0, sizeof(IRQ_queue_t));


	memset(&opp->dst[i].servicing, 0, sizeof(IRQ_queue_t));


    }

    /* Initialise timers */

    for (i = 0; i < MAX_TMR; i++) {

	opp->timers[i].ticc = 0x00000000;

	opp->timers[i].tibc = 0x80000000;

    }

    /* Initialise doorbells */

#if MAX_DBL > 0

    opp->dar = 0x00000000;

    for (i = 0; i < MAX_DBL; i++) {

	opp->doorbells[i].dmr  = 0x00000000;

    }

#endif

    /* Initialise mailboxes */

#if MAX_MBX > 0

    for (i = 0; i < MAX_MBX; i++) { /* ? */

	opp->mailboxes[i].mbr   = 0x00000000;

    }

#endif

    /* Go out of RESET state */

    opp->glbc = 0x00000000;

}