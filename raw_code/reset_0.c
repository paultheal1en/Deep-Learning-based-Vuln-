static void reset(DeviceState *d)

{

    sPAPRDRConnector *drc = SPAPR_DR_CONNECTOR(d);

    sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);



    trace_spapr_drc_reset(spapr_drc_index(drc));



    g_free(drc->ccs);

    drc->ccs = NULL;



    /* immediately upon reset we can safely assume DRCs whose devices

     * are pending removal can be safely removed, and that they will

     * subsequently be left in an ISOLATED state. move the DRC to this

     * state in these cases (which will in turn complete any pending

     * device removals)

     */

    if (drc->awaiting_release) {

        drck->set_isolation_state(drc, SPAPR_DR_ISOLATION_STATE_ISOLATED);

        /* generally this should also finalize the removal, but if the device

         * hasn't yet been configured we normally defer removal under the

         * assumption that this transition is taking place as part of device

         * configuration. so check if we're still waiting after this, and

         * force removal if we are

         */

        if (drc->awaiting_release) {

            spapr_drc_detach(drc, DEVICE(drc->dev), NULL);

        }



        /* non-PCI devices may be awaiting a transition to UNUSABLE */

        if (spapr_drc_type(drc) != SPAPR_DR_CONNECTOR_TYPE_PCI &&

            drc->awaiting_release) {

            drck->set_allocation_state(drc, SPAPR_DR_ALLOCATION_STATE_UNUSABLE);

        }

    }

}
