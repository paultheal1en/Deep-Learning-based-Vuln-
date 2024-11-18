void css_queue_crw(uint8_t rsc, uint8_t erc, int chain, uint16_t rsid)

{

    CrwContainer *crw_cont;



    trace_css_crw(rsc, erc, rsid, chain ? "(chained)" : "");

    /* TODO: Maybe use a static crw pool? */

    crw_cont = g_try_malloc0(sizeof(CrwContainer));

    if (!crw_cont) {

        channel_subsys.crws_lost = true;

        return;

    }

    crw_cont->crw.flags = (rsc << 8) | erc;

    if (chain) {

        crw_cont->crw.flags |= CRW_FLAGS_MASK_C;

    }

    crw_cont->crw.rsid = rsid;

    if (channel_subsys.crws_lost) {

        crw_cont->crw.flags |= CRW_FLAGS_MASK_R;

        channel_subsys.crws_lost = false;

    }



    QTAILQ_INSERT_TAIL(&channel_subsys.pending_crws, crw_cont, sibling);



    if (channel_subsys.do_crw_mchk) {

        channel_subsys.do_crw_mchk = false;

        /* Inject crw pending machine check. */

        s390_crw_mchk();

    }

}
