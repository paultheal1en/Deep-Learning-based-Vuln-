static uint64_t omap_mpui_read(void *opaque, target_phys_addr_t addr,

                               unsigned size)

{

    struct omap_mpu_state_s *s = (struct omap_mpu_state_s *) opaque;



    if (size != 4) {

        return omap_badwidth_read32(opaque, addr);

    }



    switch (addr) {

    case 0x00:	/* CTRL */

        return s->mpui_ctrl;

    case 0x04:	/* DEBUG_ADDR */

        return 0x01ffffff;

    case 0x08:	/* DEBUG_DATA */

        return 0xffffffff;

    case 0x0c:	/* DEBUG_FLAG */

        return 0x00000800;

    case 0x10:	/* STATUS */

        return 0x00000000;



    /* Not in OMAP310 */

    case 0x14:	/* DSP_STATUS */

    case 0x18:	/* DSP_BOOT_CONFIG */

        return 0x00000000;

    case 0x1c:	/* DSP_MPUI_CONFIG */

        return 0x0000ffff;

    }



    OMAP_BAD_REG(addr);

    return 0;

}
