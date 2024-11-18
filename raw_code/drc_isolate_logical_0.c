static uint32_t drc_isolate_logical(sPAPRDRConnector *drc)

{

    /* if the guest is configuring a device attached to this DRC, we

     * should reset the configuration state at this point since it may

     * no longer be reliable (guest released device and needs to start

     * over, or unplug occurred so the FDT is no longer valid)

     */

    g_free(drc->ccs);

    drc->ccs = NULL;



    /*

     * Fail any requests to ISOLATE the LMB DRC if this LMB doesn't

     * belong to a DIMM device that is marked for removal.

     *

     * Currently the guest userspace tool drmgr that drives the memory

     * hotplug/unplug will just try to remove a set of 'removable' LMBs

     * in response to a hot unplug request that is based on drc-count.

     * If the LMB being removed doesn't belong to a DIMM device that is

     * actually being unplugged, fail the isolation request here.

     */

    if (spapr_drc_type(drc) == SPAPR_DR_CONNECTOR_TYPE_LMB

        && !drc->unplug_requested) {

        return RTAS_OUT_HW_ERROR;

    }



    drc->isolation_state = SPAPR_DR_ISOLATION_STATE_ISOLATED;



    /* if we're awaiting release, but still in an unconfigured state,

     * it's likely the guest is still in the process of configuring

     * the device and is transitioning the devices to an ISOLATED

     * state as a part of that process. so we only complete the

     * removal when this transition happens for a device in a

     * configured state, as suggested by the state diagram from PAPR+

     * 2.7, 13.4

     */

    if (drc->unplug_requested) {

        uint32_t drc_index = spapr_drc_index(drc);

        if (drc->configured) {

            trace_spapr_drc_set_isolation_state_finalizing(drc_index);

            spapr_drc_detach(drc);

        } else {

            trace_spapr_drc_set_isolation_state_deferring(drc_index);

        }

    }

    drc->configured = false;



    return RTAS_OUT_SUCCESS;

}
