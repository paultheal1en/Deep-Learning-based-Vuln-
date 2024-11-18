static void taihu_cpld_writeb (void *opaque,

                               hwaddr addr, uint32_t value)

{

    taihu_cpld_t *cpld;



    cpld = opaque;

    switch (addr) {

    case 0x0:

        /* Read only */

        break;

    case 0x1:

        cpld->reg1 = value;

        break;

    default:

        break;

    }

}
