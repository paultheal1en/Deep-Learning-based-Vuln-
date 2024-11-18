static void pxa2xx_ssp_fifo_update(PXA2xxSSPState *s)

{

    s->sssr &= ~(0xf << 12);	/* Clear RFL */

    s->sssr &= ~(0xf << 8);	/* Clear TFL */

    s->sssr &= ~SSSR_TNF;

    if (s->enable) {

        s->sssr |= ((s->rx_level - 1) & 0xf) << 12;

        if (s->rx_level >= SSCR1_RFT(s->sscr[1]))

            s->sssr |= SSSR_RFS;

        else

            s->sssr &= ~SSSR_RFS;

        if (0 <= SSCR1_TFT(s->sscr[1]))

            s->sssr |= SSSR_TFS;

        else

            s->sssr &= ~SSSR_TFS;

        if (s->rx_level)

            s->sssr |= SSSR_RNE;

        else

            s->sssr &= ~SSSR_RNE;

        s->sssr |= SSSR_TNF;

    }



    pxa2xx_ssp_int_update(s);

}
