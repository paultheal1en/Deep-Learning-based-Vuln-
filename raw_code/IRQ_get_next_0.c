static int IRQ_get_next(OpenPICState *opp, IRQQueue *q)

{

    if (q->next == -1) {

        /* XXX: optimize */

        IRQ_check(opp, q);

    }



    return q->next;

}
