static void xics_reset(void *opaque)

{

    struct icp_state *icp = (struct icp_state *)opaque;

    struct ics_state *ics = icp->ics;

    int i;



    for (i = 0; i < icp->nr_servers; i++) {

        icp->ss[i].xirr = 0;

        icp->ss[i].pending_priority = 0;

        icp->ss[i].mfrr = 0xff;

        /* Make all outputs are deasserted */

        qemu_set_irq(icp->ss[i].output, 0);

    }



    for (i = 0; i < ics->nr_irqs; i++) {

        /* Reset everything *except* the type */

        ics->irqs[i].server = 0;

        ics->irqs[i].status = 0;

        ics->irqs[i].priority = 0xff;

        ics->irqs[i].saved_priority = 0xff;

    }

}
