static uint32_t drc_isolate_physical(sPAPRDRConnector *drc)

{

    switch (drc->state) {

    case SPAPR_DRC_STATE_PHYSICAL_POWERON:

        return RTAS_OUT_SUCCESS; /* Nothing to do */

    case SPAPR_DRC_STATE_PHYSICAL_CONFIGURED:

        break; /* see below */

    case SPAPR_DRC_STATE_PHYSICAL_UNISOLATE:

        return RTAS_OUT_PARAM_ERROR; /* not allowed */

    default:

        g_assert_not_reached();

    }



    /* if the guest is configuring a device attached to this DRC, we

     * should reset the configuration state at this point since it may

     * no longer be reliable (guest released device and needs to start

     * over, or unplug occurred so the FDT is no longer valid)

     */

    g_free(drc->ccs);

    drc->ccs = NULL;



    drc->state = SPAPR_DRC_STATE_PHYSICAL_POWERON;



    if (drc->unplug_requested) {

        uint32_t drc_index = spapr_drc_index(drc);

        trace_spapr_drc_set_isolation_state_finalizing(drc_index);

        spapr_drc_detach(drc);

    }



    return RTAS_OUT_SUCCESS;

}
