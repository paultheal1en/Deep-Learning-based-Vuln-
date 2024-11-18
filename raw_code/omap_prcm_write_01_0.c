static void omap_prcm_write(void *opaque, hwaddr addr,

                            uint64_t value, unsigned size)

{

    struct omap_prcm_s *s = (struct omap_prcm_s *) opaque;



    if (size != 4) {

        omap_badwidth_write32(opaque, addr, value);

        return;

    }



    switch (addr) {

    case 0x000:	/* PRCM_REVISION */

    case 0x054:	/* PRCM_VOLTST */

    case 0x084:	/* PRCM_CLKCFG_STATUS */

    case 0x1e4:	/* PM_PWSTST_MPU */

    case 0x220:	/* CM_IDLEST1_CORE */

    case 0x224:	/* CM_IDLEST2_CORE */

    case 0x22c:	/* CM_IDLEST4_CORE */

    case 0x2c8:	/* PM_WKDEP_CORE */

    case 0x2e4:	/* PM_PWSTST_CORE */

    case 0x320:	/* CM_IDLEST_GFX */

    case 0x3e4:	/* PM_PWSTST_GFX */

    case 0x420:	/* CM_IDLEST_WKUP */

    case 0x520:	/* CM_IDLEST_CKGEN */

    case 0x820:	/* CM_IDLEST_DSP */

    case 0x8e4:	/* PM_PWSTST_DSP */

        OMAP_RO_REG(addr);

        return;



    case 0x010:	/* PRCM_SYSCONFIG */

        s->sysconfig = value & 1;

        break;



    case 0x018:	/* PRCM_IRQSTATUS_MPU */

        s->irqst[0] &= ~value;

        omap_prcm_int_update(s, 0);

        break;

    case 0x01c:	/* PRCM_IRQENABLE_MPU */

        s->irqen[0] = value & 0x3f;

        omap_prcm_int_update(s, 0);

        break;



    case 0x050:	/* PRCM_VOLTCTRL */

        s->voltctrl = value & 0xf1c3;

        break;



    case 0x060:	/* PRCM_CLKSRC_CTRL */

        s->clksrc[0] = value & 0xdb;

        /* TODO update clocks */

        break;



    case 0x070:	/* PRCM_CLKOUT_CTRL */

        s->clkout[0] = value & 0xbbbb;

        /* TODO update clocks */

        break;



    case 0x078:	/* PRCM_CLKEMUL_CTRL */

        s->clkemul[0] = value & 1;

        /* TODO update clocks */

        break;



    case 0x080:	/* PRCM_CLKCFG_CTRL */

        break;



    case 0x090:	/* PRCM_VOLTSETUP */

        s->setuptime[0] = value & 0xffff;

        break;

    case 0x094:	/* PRCM_CLKSSETUP */

        s->setuptime[1] = value & 0xffff;

        break;



    case 0x098:	/* PRCM_POLCTRL */

        s->clkpol[0] = value & 0x701;

        break;



    case 0x0b0:	/* GENERAL_PURPOSE1 */

    case 0x0b4:	/* GENERAL_PURPOSE2 */

    case 0x0b8:	/* GENERAL_PURPOSE3 */

    case 0x0bc:	/* GENERAL_PURPOSE4 */

    case 0x0c0:	/* GENERAL_PURPOSE5 */

    case 0x0c4:	/* GENERAL_PURPOSE6 */

    case 0x0c8:	/* GENERAL_PURPOSE7 */

    case 0x0cc:	/* GENERAL_PURPOSE8 */

    case 0x0d0:	/* GENERAL_PURPOSE9 */

    case 0x0d4:	/* GENERAL_PURPOSE10 */

    case 0x0d8:	/* GENERAL_PURPOSE11 */

    case 0x0dc:	/* GENERAL_PURPOSE12 */

    case 0x0e0:	/* GENERAL_PURPOSE13 */

    case 0x0e4:	/* GENERAL_PURPOSE14 */

    case 0x0e8:	/* GENERAL_PURPOSE15 */

    case 0x0ec:	/* GENERAL_PURPOSE16 */

    case 0x0f0:	/* GENERAL_PURPOSE17 */

    case 0x0f4:	/* GENERAL_PURPOSE18 */

    case 0x0f8:	/* GENERAL_PURPOSE19 */

    case 0x0fc:	/* GENERAL_PURPOSE20 */

        s->scratch[(addr - 0xb0) >> 2] = value;

        break;



    case 0x140:	/* CM_CLKSEL_MPU */

        s->clksel[0] = value & 0x1f;

        /* TODO update clocks */

        break;

    case 0x148:	/* CM_CLKSTCTRL_MPU */

        s->clkctrl[0] = value & 0x1f;

        break;



    case 0x158:	/* RM_RSTST_MPU */

        s->rst[0] &= ~value;

        break;

    case 0x1c8:	/* PM_WKDEP_MPU */

        s->wkup[0] = value & 0x15;

        break;



    case 0x1d4:	/* PM_EVGENCTRL_MPU */

        s->ev = value & 0x1f;

        break;

    case 0x1d8:	/* PM_EVEGENONTIM_MPU */

        s->evtime[0] = value;

        break;

    case 0x1dc:	/* PM_EVEGENOFFTIM_MPU */

        s->evtime[1] = value;

        break;



    case 0x1e0:	/* PM_PWSTCTRL_MPU */

        s->power[0] = value & 0xc0f;

        break;



    case 0x200:	/* CM_FCLKEN1_CORE */

        s->clken[0] = value & 0xbfffffff;

        /* TODO update clocks */

        /* The EN_EAC bit only gets/puts func_96m_clk.  */

        break;

    case 0x204:	/* CM_FCLKEN2_CORE */

        s->clken[1] = value & 0x00000007;

        /* TODO update clocks */

        break;

    case 0x210:	/* CM_ICLKEN1_CORE */

        s->clken[2] = value & 0xfffffff9;

        /* TODO update clocks */

        /* The EN_EAC bit only gets/puts core_l4_iclk.  */

        break;

    case 0x214:	/* CM_ICLKEN2_CORE */

        s->clken[3] = value & 0x00000007;

        /* TODO update clocks */

        break;

    case 0x21c:	/* CM_ICLKEN4_CORE */

        s->clken[4] = value & 0x0000001f;

        /* TODO update clocks */

        break;



    case 0x230:	/* CM_AUTOIDLE1_CORE */

        s->clkidle[0] = value & 0xfffffff9;

        /* TODO update clocks */

        break;

    case 0x234:	/* CM_AUTOIDLE2_CORE */

        s->clkidle[1] = value & 0x00000007;

        /* TODO update clocks */

        break;

    case 0x238:	/* CM_AUTOIDLE3_CORE */

        s->clkidle[2] = value & 0x00000007;

        /* TODO update clocks */

        break;

    case 0x23c:	/* CM_AUTOIDLE4_CORE */

        s->clkidle[3] = value & 0x0000001f;

        /* TODO update clocks */

        break;



    case 0x240:	/* CM_CLKSEL1_CORE */

        s->clksel[1] = value & 0x0fffbf7f;

        /* TODO update clocks */

        break;



    case 0x244:	/* CM_CLKSEL2_CORE */

        s->clksel[2] = value & 0x00fffffc;

        /* TODO update clocks */

        break;



    case 0x248:	/* CM_CLKSTCTRL_CORE */

        s->clkctrl[1] = value & 0x7;

        break;



    case 0x2a0:	/* PM_WKEN1_CORE */

        s->wken[0] = value & 0x04667ff8;

        break;

    case 0x2a4:	/* PM_WKEN2_CORE */

        s->wken[1] = value & 0x00000005;

        break;



    case 0x2b0:	/* PM_WKST1_CORE */

        s->wkst[0] &= ~value;

        break;

    case 0x2b4:	/* PM_WKST2_CORE */

        s->wkst[1] &= ~value;

        break;



    case 0x2e0:	/* PM_PWSTCTRL_CORE */

        s->power[1] = (value & 0x00fc3f) | (1 << 2);

        break;



    case 0x300:	/* CM_FCLKEN_GFX */

        s->clken[5] = value & 6;

        /* TODO update clocks */

        break;

    case 0x310:	/* CM_ICLKEN_GFX */

        s->clken[6] = value & 1;

        /* TODO update clocks */

        break;

    case 0x340:	/* CM_CLKSEL_GFX */

        s->clksel[3] = value & 7;

        /* TODO update clocks */

        break;

    case 0x348:	/* CM_CLKSTCTRL_GFX */

        s->clkctrl[2] = value & 1;

        break;

    case 0x350:	/* RM_RSTCTRL_GFX */

        s->rstctrl[0] = value & 1;

        /* TODO: reset */

        break;

    case 0x358:	/* RM_RSTST_GFX */

        s->rst[1] &= ~value;

        break;

    case 0x3c8:	/* PM_WKDEP_GFX */

        s->wkup[1] = value & 0x13;

        break;

    case 0x3e0:	/* PM_PWSTCTRL_GFX */

        s->power[2] = (value & 0x00c0f) | (3 << 2);

        break;



    case 0x400:	/* CM_FCLKEN_WKUP */

        s->clken[7] = value & 0xd;

        /* TODO update clocks */

        break;

    case 0x410:	/* CM_ICLKEN_WKUP */

        s->clken[8] = value & 0x3f;

        /* TODO update clocks */

        break;

    case 0x430:	/* CM_AUTOIDLE_WKUP */

        s->clkidle[4] = value & 0x0000003f;

        /* TODO update clocks */

        break;

    case 0x440:	/* CM_CLKSEL_WKUP */

        s->clksel[4] = value & 3;

        /* TODO update clocks */

        break;

    case 0x450:	/* RM_RSTCTRL_WKUP */

        /* TODO: reset */

        if (value & 2)

            qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET);

        break;

    case 0x454:	/* RM_RSTTIME_WKUP */

        s->rsttime_wkup = value & 0x1fff;

        break;

    case 0x458:	/* RM_RSTST_WKUP */

        s->rst[2] &= ~value;

        break;

    case 0x4a0:	/* PM_WKEN_WKUP */

        s->wken[2] = value & 0x00000005;

        break;

    case 0x4b0:	/* PM_WKST_WKUP */

        s->wkst[2] &= ~value;

        break;



    case 0x500:	/* CM_CLKEN_PLL */

        if (value & 0xffffff30)

            fprintf(stderr, "%s: write 0s in CM_CLKEN_PLL for "

                            "future compatibility\n", __FUNCTION__);

        if ((s->clken[9] ^ value) & 0xcc) {

            s->clken[9] &= ~0xcc;

            s->clken[9] |= value & 0xcc;

            omap_prcm_apll_update(s);

        }

        if ((s->clken[9] ^ value) & 3) {

            s->clken[9] &= ~3;

            s->clken[9] |= value & 3;

            omap_prcm_dpll_update(s);

        }

        break;

    case 0x530:	/* CM_AUTOIDLE_PLL */

        s->clkidle[5] = value & 0x000000cf;

        /* TODO update clocks */

        break;

    case 0x540:	/* CM_CLKSEL1_PLL */

        if (value & 0xfc4000d7)

            fprintf(stderr, "%s: write 0s in CM_CLKSEL1_PLL for "

                            "future compatibility\n", __FUNCTION__);

        if ((s->clksel[5] ^ value) & 0x003fff00) {

            s->clksel[5] = value & 0x03bfff28;

            omap_prcm_dpll_update(s);

        }

        /* TODO update the other clocks */



        s->clksel[5] = value & 0x03bfff28;

        break;

    case 0x544:	/* CM_CLKSEL2_PLL */

        if (value & ~3)

            fprintf(stderr, "%s: write 0s in CM_CLKSEL2_PLL[31:2] for "

                            "future compatibility\n", __FUNCTION__);

        if (s->clksel[6] != (value & 3)) {

            s->clksel[6] = value & 3;

            omap_prcm_dpll_update(s);

        }

        break;



    case 0x800:	/* CM_FCLKEN_DSP */

        s->clken[10] = value & 0x501;

        /* TODO update clocks */

        break;

    case 0x810:	/* CM_ICLKEN_DSP */

        s->clken[11] = value & 0x2;

        /* TODO update clocks */

        break;

    case 0x830:	/* CM_AUTOIDLE_DSP */

        s->clkidle[6] = value & 0x2;

        /* TODO update clocks */

        break;

    case 0x840:	/* CM_CLKSEL_DSP */

        s->clksel[7] = value & 0x3fff;

        /* TODO update clocks */

        break;

    case 0x848:	/* CM_CLKSTCTRL_DSP */

        s->clkctrl[3] = value & 0x101;

        break;

    case 0x850:	/* RM_RSTCTRL_DSP */

        /* TODO: reset */

        break;

    case 0x858:	/* RM_RSTST_DSP */

        s->rst[3] &= ~value;

        break;

    case 0x8c8:	/* PM_WKDEP_DSP */

        s->wkup[2] = value & 0x13;

        break;

    case 0x8e0:	/* PM_PWSTCTRL_DSP */

        s->power[3] = (value & 0x03017) | (3 << 2);

        break;



    case 0x8f0:	/* PRCM_IRQSTATUS_DSP */

        s->irqst[1] &= ~value;

        omap_prcm_int_update(s, 1);

        break;

    case 0x8f4:	/* PRCM_IRQENABLE_DSP */

        s->irqen[1] = value & 0x7;

        omap_prcm_int_update(s, 1);

        break;



    case 0x8f8:	/* PRCM_IRQSTATUS_IVA */

        s->irqst[2] &= ~value;

        omap_prcm_int_update(s, 2);

        break;

    case 0x8fc:	/* PRCM_IRQENABLE_IVA */

        s->irqen[2] = value & 0x7;

        omap_prcm_int_update(s, 2);

        break;



    default:

        OMAP_BAD_REG(addr);

        return;

    }

}
