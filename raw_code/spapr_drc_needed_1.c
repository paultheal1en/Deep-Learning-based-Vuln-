static bool spapr_drc_needed(void *opaque)

{

    sPAPRDRConnector *drc = (sPAPRDRConnector *)opaque;

    sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);



    /* If no dev is plugged in there is no need to migrate the DRC state */

    if (!drc->dev) {

        return false;

    }



    /*

     * We need to migrate the state if it's not equal to the expected

     * long-term state, which is the same as the coldplugged initial

     * state */

    return (drc->state != drck->ready_state);

}
