static void omap_clkm_write(void *opaque, target_phys_addr_t addr,

                            uint64_t value, unsigned size)

{

    struct omap_mpu_state_s *s = (struct omap_mpu_state_s *) opaque;

    uint16_t diff;

    omap_clk clk;

    static const char *clkschemename[8] = {

        "fully synchronous", "fully asynchronous", "synchronous scalable",

        "mix mode 1", "mix mode 2", "bypass mode", "mix mode 3", "mix mode 4",

    };



    if (size != 2) {

        return omap_badwidth_write16(opaque, addr, value);

    }



    switch (addr) {

    case 0x00:	/* ARM_CKCTL */

        diff = s->clkm.arm_ckctl ^ value;

        s->clkm.arm_ckctl = value & 0x7fff;

        omap_clkm_ckctl_update(s, diff, value);

        return;



    case 0x04:	/* ARM_IDLECT1 */

        diff = s->clkm.arm_idlect1 ^ value;

        s->clkm.arm_idlect1 = value & 0x0fff;

        omap_clkm_idlect1_update(s, diff, value);

        return;



    case 0x08:	/* ARM_IDLECT2 */

        diff = s->clkm.arm_idlect2 ^ value;

        s->clkm.arm_idlect2 = value & 0x07ff;

        omap_clkm_idlect2_update(s, diff, value);

        return;



    case 0x0c:	/* ARM_EWUPCT */

        s->clkm.arm_ewupct = value & 0x003f;

        return;



    case 0x10:	/* ARM_RSTCT1 */

        diff = s->clkm.arm_rstct1 ^ value;

        s->clkm.arm_rstct1 = value & 0x0007;

        if (value & 9) {

            qemu_system_reset_request();

            s->clkm.cold_start = 0xa;

        }

        if (diff & ~value & 4) {				/* DSP_RST */

            omap_mpui_reset(s);

            omap_tipb_bridge_reset(s->private_tipb);

            omap_tipb_bridge_reset(s->public_tipb);

        }

        if (diff & 2) {						/* DSP_EN */

            clk = omap_findclk(s, "dsp_ck");

            omap_clk_canidle(clk, (~value >> 1) & 1);

        }

        return;



    case 0x14:	/* ARM_RSTCT2 */

        s->clkm.arm_rstct2 = value & 0x0001;

        return;



    case 0x18:	/* ARM_SYSST */

        if ((s->clkm.clocking_scheme ^ (value >> 11)) & 7) {

            s->clkm.clocking_scheme = (value >> 11) & 7;

            printf("%s: clocking scheme set to %s\n", __FUNCTION__,

                            clkschemename[s->clkm.clocking_scheme]);

        }

        s->clkm.cold_start &= value & 0x3f;

        return;



    case 0x1c:	/* ARM_CKOUT1 */

        diff = s->clkm.arm_ckout1 ^ value;

        s->clkm.arm_ckout1 = value & 0x003f;

        omap_clkm_ckout1_update(s, diff, value);

        return;



    case 0x20:	/* ARM_CKOUT2 */

    default:

        OMAP_BAD_REG(addr);

    }

}
