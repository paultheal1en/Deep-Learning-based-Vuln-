int spapr_allocate_irq_block(int num, bool lsi)

{

    int first = -1;

    int i;



    for (i = 0; i < num; ++i) {

        int irq;



        irq = spapr_allocate_irq(0, lsi);

        if (!irq) {

            return -1;

        }



        if (0 == i) {

            first = irq;

        }



        /* If the above doesn't create a consecutive block then that's

         * an internal bug */

        assert(irq == (first + i));

    }



    return first;

}
