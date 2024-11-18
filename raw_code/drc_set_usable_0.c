static uint32_t drc_set_usable(sPAPRDRConnector *drc)

{

    /* if there's no resource/device associated with the DRC, there's

     * no way for us to put it in an allocation state consistent with

     * being 'USABLE'. PAPR 2.7, 13.5.3.4 documents that this should

     * result in an RTAS return code of -3 / "no such indicator"

     */

    if (!drc->dev) {

        return RTAS_OUT_NO_SUCH_INDICATOR;

    }

    if (drc->awaiting_release) {

        /* Don't allow the guest to move a device away from UNUSABLE

         * state when we want to unplug it */

        return RTAS_OUT_NO_SUCH_INDICATOR;

    }



    drc->allocation_state = SPAPR_DR_ALLOCATION_STATE_USABLE;



    return RTAS_OUT_SUCCESS;

}
