static void spapr_hotplug_set_signalled(uint32_t drc_index)

{

    sPAPRDRConnector *drc = spapr_drc_by_index(drc_index);

    sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

    drck->set_signalled(drc);

}
