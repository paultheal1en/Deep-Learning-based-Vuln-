static uint64_t exynos4210_i2c_read(void *opaque, target_phys_addr_t offset,

                                 unsigned size)

{

    Exynos4210I2CState *s = (Exynos4210I2CState *)opaque;

    uint8_t value;



    switch (offset) {

    case I2CCON_ADDR:

        value = s->i2ccon;

        break;

    case I2CSTAT_ADDR:

        value = s->i2cstat;

        break;

    case I2CADD_ADDR:

        value = s->i2cadd;

        break;

    case I2CDS_ADDR:

        value = s->i2cds;

        s->scl_free = true;

        if (EXYNOS4_I2C_MODE(s->i2cstat) == I2CMODE_MASTER_Rx &&

               (s->i2cstat & I2CSTAT_START_BUSY) &&

               !(s->i2ccon & I2CCON_INT_PEND)) {

            exynos4210_i2c_data_receive(s);

        }

        break;

    case I2CLC_ADDR:

        value = s->i2clc;

        break;

    default:

        value = 0;

        DPRINT("ERROR: Bad read offset 0x%x\n", (unsigned int)offset);

        break;

    }



    DPRINT("read %s [0x%02x] -> 0x%02x\n", exynos4_i2c_get_regname(offset),

            (unsigned int)offset, value);

    return value;

}
