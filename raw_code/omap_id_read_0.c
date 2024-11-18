static uint64_t omap_id_read(void *opaque, target_phys_addr_t addr,

                             unsigned size)

{

    struct omap_mpu_state_s *s = (struct omap_mpu_state_s *) opaque;



    if (size != 4) {

        return omap_badwidth_read32(opaque, addr);

    }



    switch (addr) {

    case 0xfffe1800:	/* DIE_ID_LSB */

        return 0xc9581f0e;

    case 0xfffe1804:	/* DIE_ID_MSB */

        return 0xa8858bfa;



    case 0xfffe2000:	/* PRODUCT_ID_LSB */

        return 0x00aaaafc;

    case 0xfffe2004:	/* PRODUCT_ID_MSB */

        return 0xcafeb574;



    case 0xfffed400:	/* JTAG_ID_LSB */

        switch (s->mpu_model) {

        case omap310:

            return 0x03310315;

        case omap1510:

            return 0x03310115;

        default:

            hw_error("%s: bad mpu model\n", __FUNCTION__);

        }

        break;



    case 0xfffed404:	/* JTAG_ID_MSB */

        switch (s->mpu_model) {

        case omap310:

            return 0xfb57402f;

        case omap1510:

            return 0xfb47002f;

        default:

            hw_error("%s: bad mpu model\n", __FUNCTION__);

        }

        break;

    }



    OMAP_BAD_REG(addr);

    return 0;

}
