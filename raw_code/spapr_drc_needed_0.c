static bool spapr_drc_needed(void *opaque)

{

    sPAPRDRConnector *drc = (sPAPRDRConnector *)opaque;

    sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

    bool rc = false;

    sPAPRDREntitySense value = drck->dr_entity_sense(drc);



    /* If no dev is plugged in there is no need to migrate the DRC state */

    if (value != SPAPR_DR_ENTITY_SENSE_PRESENT) {

        return false;

    }



    /*

     * If there is dev plugged in, we need to migrate the DRC state when

     * it is different from cold-plugged state

     */

    switch (spapr_drc_type(drc)) {

    case SPAPR_DR_CONNECTOR_TYPE_PCI:

    case SPAPR_DR_CONNECTOR_TYPE_CPU:

    case SPAPR_DR_CONNECTOR_TYPE_LMB:

        rc = !((drc->isolation_state == SPAPR_DR_ISOLATION_STATE_UNISOLATED) &&

               (drc->allocation_state == SPAPR_DR_ALLOCATION_STATE_USABLE) &&

               drc->configured && !drc->awaiting_release);

        break;

    case SPAPR_DR_CONNECTOR_TYPE_PHB:

    case SPAPR_DR_CONNECTOR_TYPE_VIO:

    default:

        g_assert_not_reached();

    }

    return rc;

}
