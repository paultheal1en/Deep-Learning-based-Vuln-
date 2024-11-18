static void exynos4210_fimd_reset(DeviceState *d)

{

    Exynos4210fimdState *s = EXYNOS4210_FIMD(d);

    unsigned w;



    DPRINT_TRACE("Display controller reset\n");

    /* Set all display controller registers to 0 */

    memset(&s->vidcon, 0, (uint8_t *)&s->window - (uint8_t *)&s->vidcon);

    for (w = 0; w < NUM_OF_WINDOWS; w++) {

        memset(&s->window[w], 0, sizeof(Exynos4210fimdWindow));

        s->window[w].blendeq = 0xC2;

        exynos4210_fimd_update_win_bppmode(s, w);

        exynos4210_fimd_trace_bppmode(s, w, 0xFFFFFFFF);

        fimd_update_get_alpha(s, w);

    }



    if (s->ifb != NULL) {

        g_free(s->ifb);

    }

    s->ifb = NULL;



    exynos4210_fimd_invalidate(s);

    exynos4210_fimd_enable(s, false);

    /* Some registers have non-zero initial values */

    s->winchmap = 0x7D517D51;

    s->colorgaincon = 0x10040100;

    s->huecoef_cr[0] = s->huecoef_cr[3] = 0x01000100;

    s->huecoef_cb[0] = s->huecoef_cb[3] = 0x01000100;

    s->hueoffset = 0x01800080;

}
