static void omap_prcm_apll_update(struct omap_prcm_s *s)

{

    int mode[2];



    mode[0] = (s->clken[9] >> 6) & 3;

    s->apll_lock[0] = (mode[0] == 3);

    mode[1] = (s->clken[9] >> 2) & 3;

    s->apll_lock[1] = (mode[1] == 3);

    /* TODO: update clocks */



    if (mode[0] == 1 || mode[0] == 2 || mode[1] == 1 || mode[1] == 2)

        fprintf(stderr, "%s: bad EN_54M_PLL or bad EN_96M_PLL\n",

                        __FUNCTION__);

}
