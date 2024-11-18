static bool nvic_rettobase(NVICState *s)

{

    int irq, nhand = 0;



    for (irq = ARMV7M_EXCP_RESET; irq < s->num_irq; irq++) {

        if (s->vectors[irq].active) {

            nhand++;

            if (nhand == 2) {

                return 0;

            }

        }

    }



    return 1;

}
