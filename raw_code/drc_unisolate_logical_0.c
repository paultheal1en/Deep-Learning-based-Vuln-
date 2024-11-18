static uint32_t drc_unisolate_logical(sPAPRDRConnector *drc)

{

    /* cannot unisolate a non-existent resource, and, or resources

     * which are in an 'UNUSABLE' allocation state. (PAPR 2.7,

     * 13.5.3.5)

     */

    if (!drc->dev ||

        drc->allocation_state == SPAPR_DR_ALLOCATION_STATE_UNUSABLE) {

        return RTAS_OUT_NO_SUCH_INDICATOR;

    }



    drc->isolation_state = SPAPR_DR_ISOLATION_STATE_UNISOLATED;



    return RTAS_OUT_SUCCESS;

}
