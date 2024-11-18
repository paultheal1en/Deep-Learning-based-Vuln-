static void rtas_set_xive(sPAPREnvironment *spapr, uint32_t token,

                          uint32_t nargs, target_ulong args,

                          uint32_t nret, target_ulong rets)

{

    struct ics_state *ics = spapr->icp->ics;

    uint32_t nr, server, priority;



    if ((nargs != 3) || (nret != 1)) {

        rtas_st(rets, 0, -3);

        return;

    }



    nr = rtas_ld(args, 0);

    server = rtas_ld(args, 1);

    priority = rtas_ld(args, 2);



    if (!ics_valid_irq(ics, nr) || (server >= ics->icp->nr_servers)

        || (priority > 0xff)) {

        rtas_st(rets, 0, -3);

        return;

    }



    ics_write_xive(ics, nr, server, priority, priority);



    rtas_st(rets, 0, 0); /* Success */

}
