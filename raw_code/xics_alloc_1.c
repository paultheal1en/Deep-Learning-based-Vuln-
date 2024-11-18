int xics_alloc(XICSState *icp, int src, int irq_hint, bool lsi)

{

    ICSState *ics = &icp->ics[src];

    int irq;



    if (irq_hint) {

        assert(src == xics_find_source(icp, irq_hint));

        if (!ICS_IRQ_FREE(ics, irq_hint - ics->offset)) {

            trace_xics_alloc_failed_hint(src, irq_hint);

            return -1;

        }

        irq = irq_hint;

    } else {

        irq = ics_find_free_block(ics, 1, 1);

        if (irq < 0) {

            trace_xics_alloc_failed_no_left(src);

            return -1;

        }

        irq += ics->offset;

    }



    ics_set_irq_type(ics, irq - ics->offset, lsi);

    trace_xics_alloc(src, irq);



    return irq;

}
