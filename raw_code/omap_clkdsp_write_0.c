static void omap_clkdsp_write(void *opaque, target_phys_addr_t addr,

                              uint64_t value, unsigned size)

{

    struct omap_mpu_state_s *s = (struct omap_mpu_state_s *) opaque;

    uint16_t diff;



    if (size != 2) {

        return omap_badwidth_write16(opaque, addr, value);

    }



    switch (addr) {

    case 0x04:	/* DSP_IDLECT1 */

        diff = s->clkm.dsp_idlect1 ^ value;

        s->clkm.dsp_idlect1 = value & 0x01f7;

        omap_clkdsp_idlect1_update(s, diff, value);

        break;



    case 0x08:	/* DSP_IDLECT2 */

        s->clkm.dsp_idlect2 = value & 0x0037;

        diff = s->clkm.dsp_idlect1 ^ value;

        omap_clkdsp_idlect2_update(s, diff, value);

        break;



    case 0x14:	/* DSP_RSTCT2 */

        s->clkm.dsp_rstct2 = value & 0x0001;

        break;



    case 0x18:	/* DSP_SYSST */

        s->clkm.cold_start &= value & 0x3f;

        break;



    default:

        OMAP_BAD_REG(addr);

    }

}
