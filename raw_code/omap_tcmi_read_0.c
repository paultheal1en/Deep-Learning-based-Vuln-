static uint64_t omap_tcmi_read(void *opaque, target_phys_addr_t addr,

                               unsigned size)

{

    struct omap_mpu_state_s *s = (struct omap_mpu_state_s *) opaque;

    uint32_t ret;



    if (size != 4) {

        return omap_badwidth_read32(opaque, addr);

    }



    switch (addr) {

    case 0x00:	/* IMIF_PRIO */

    case 0x04:	/* EMIFS_PRIO */

    case 0x08:	/* EMIFF_PRIO */

    case 0x0c:	/* EMIFS_CONFIG */

    case 0x10:	/* EMIFS_CS0_CONFIG */

    case 0x14:	/* EMIFS_CS1_CONFIG */

    case 0x18:	/* EMIFS_CS2_CONFIG */

    case 0x1c:	/* EMIFS_CS3_CONFIG */

    case 0x24:	/* EMIFF_MRS */

    case 0x28:	/* TIMEOUT1 */

    case 0x2c:	/* TIMEOUT2 */

    case 0x30:	/* TIMEOUT3 */

    case 0x3c:	/* EMIFF_SDRAM_CONFIG_2 */

    case 0x40:	/* EMIFS_CFG_DYN_WAIT */

        return s->tcmi_regs[addr >> 2];



    case 0x20:	/* EMIFF_SDRAM_CONFIG */

        ret = s->tcmi_regs[addr >> 2];

        s->tcmi_regs[addr >> 2] &= ~1; /* XXX: Clear SLRF on SDRAM access */

        /* XXX: We can try using the VGA_DIRTY flag for this */

        return ret;

    }



    OMAP_BAD_REG(addr);

    return 0;

}
