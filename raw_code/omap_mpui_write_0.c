static void omap_mpui_write(void *opaque, target_phys_addr_t addr,

                            uint64_t value, unsigned size)

{

    struct omap_mpu_state_s *s = (struct omap_mpu_state_s *) opaque;



    if (size != 4) {

        return omap_badwidth_write32(opaque, addr, value);

    }



    switch (addr) {

    case 0x00:	/* CTRL */

        s->mpui_ctrl = value & 0x007fffff;

        break;



    case 0x04:	/* DEBUG_ADDR */

    case 0x08:	/* DEBUG_DATA */

    case 0x0c:	/* DEBUG_FLAG */

    case 0x10:	/* STATUS */

    /* Not in OMAP310 */

    case 0x14:	/* DSP_STATUS */

        OMAP_RO_REG(addr);

    case 0x18:	/* DSP_BOOT_CONFIG */

    case 0x1c:	/* DSP_MPUI_CONFIG */

        break;



    default:

        OMAP_BAD_REG(addr);

    }

}
