static uint64_t exynos4210_mct_read(void *opaque, target_phys_addr_t offset,

        unsigned size)

{

    Exynos4210MCTState *s = (Exynos4210MCTState *)opaque;

    int index;

    int shift;

    uint64_t count;

    uint32_t value;

    int lt_i;



    switch (offset) {



    case MCT_CFG:

        value = s->reg_mct_cfg;

        break;



    case G_CNT_L: case G_CNT_U:

        shift = 8 * (offset & 0x4);

        count = exynos4210_gfrc_get_count(&s->g_timer);

        value = UINT32_MAX & (count >> shift);

        DPRINTF("read FRC=0x%llx\n", count);

        break;



    case G_CNT_WSTAT:

        value = s->g_timer.reg.cnt_wstat;

        break;



    case G_COMP_L(0): case G_COMP_L(1): case G_COMP_L(2): case G_COMP_L(3):

    case G_COMP_U(0): case G_COMP_U(1): case G_COMP_U(2): case G_COMP_U(3):

    index = GET_G_COMP_IDX(offset);

    shift = 8 * (offset & 0x4);

    value = UINT32_MAX & (s->g_timer.reg.comp[index] >> shift);

    break;



    case G_TCON:

        value = s->g_timer.reg.tcon;

        break;



    case G_INT_CSTAT:

        value = s->g_timer.reg.int_cstat;

        break;



    case G_INT_ENB:

        value = s->g_timer.reg.int_enb;

        break;

        break;

    case G_WSTAT:

        value = s->g_timer.reg.wstat;

        break;



    case G_COMP0_ADD_INCR: case G_COMP1_ADD_INCR:

    case G_COMP2_ADD_INCR: case G_COMP3_ADD_INCR:

        value = s->g_timer.reg.comp_add_incr[GET_G_COMP_ADD_INCR_IDX(offset)];

        break;



        /* Local timers */

    case L0_TCNTB: case L0_ICNTB: case L0_FRCNTB:

    case L1_TCNTB: case L1_ICNTB: case L1_FRCNTB:

        lt_i = GET_L_TIMER_IDX(offset);

        index = GET_L_TIMER_CNT_REG_IDX(offset, lt_i);

        value = s->l_timer[lt_i].reg.cnt[index];

        break;



    case L0_TCNTO: case L1_TCNTO:

        lt_i = GET_L_TIMER_IDX(offset);



        value = exynos4210_ltick_cnt_get_cnto(&s->l_timer[lt_i].tick_timer);

        DPRINTF("local timer[%d] read TCNTO %x\n", lt_i, value);

        break;



    case L0_ICNTO: case L1_ICNTO:

        lt_i = GET_L_TIMER_IDX(offset);



        value = exynos4210_ltick_int_get_cnto(&s->l_timer[lt_i].tick_timer);

        DPRINTF("local timer[%d] read ICNTO %x\n", lt_i, value);

        break;



    case L0_FRCNTO: case L1_FRCNTO:

        lt_i = GET_L_TIMER_IDX(offset);



        value = exynos4210_lfrc_get_count(&s->l_timer[lt_i]);



        break;



    case L0_TCON: case L1_TCON:

        lt_i = ((offset & 0xF00) - L0_TCNTB) / 0x100;

        value = s->l_timer[lt_i].reg.tcon;

        break;



    case L0_INT_CSTAT: case L1_INT_CSTAT:

        lt_i = ((offset & 0xF00) - L0_TCNTB) / 0x100;

        value = s->l_timer[lt_i].reg.int_cstat;

        break;



    case L0_INT_ENB: case L1_INT_ENB:

        lt_i = ((offset & 0xF00) - L0_TCNTB) / 0x100;

        value = s->l_timer[lt_i].reg.int_enb;

        break;



    case L0_WSTAT: case L1_WSTAT:

        lt_i = ((offset & 0xF00) - L0_TCNTB) / 0x100;

        value = s->l_timer[lt_i].reg.wstat;

        break;



    default:

        hw_error("exynos4210.mct: bad read offset "

                TARGET_FMT_plx "\n", offset);

        break;

    }

    return value;

}
