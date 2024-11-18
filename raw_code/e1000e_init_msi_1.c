e1000e_init_msi(E1000EState *s)

{

    int res;



    res = msi_init(PCI_DEVICE(s),

                   0xD0,   /* MSI capability offset              */

                   1,      /* MAC MSI interrupts                 */

                   true,   /* 64-bit message addresses supported */

                   false); /* Per vector mask supported          */



    if (res > 0) {

        s->intr_state |= E1000E_USE_MSI;

    } else {

        trace_e1000e_msi_init_fail(res);

    }

}
