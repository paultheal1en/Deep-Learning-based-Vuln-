static void spapr_dr_connector_class_init(ObjectClass *k, void *data)

{

    DeviceClass *dk = DEVICE_CLASS(k);

    sPAPRDRConnectorClass *drck = SPAPR_DR_CONNECTOR_CLASS(k);



    dk->reset = reset;

    dk->realize = realize;

    dk->unrealize = unrealize;

    drck->set_isolation_state = set_isolation_state;

    drck->set_indicator_state = set_indicator_state;

    drck->set_allocation_state = set_allocation_state;

    drck->get_index = get_index;

    drck->get_type = get_type;

    drck->get_name = get_name;

    drck->get_fdt = get_fdt;

    drck->set_configured = set_configured;

    drck->entity_sense = entity_sense;

    drck->attach = attach;

    drck->detach = detach;

    drck->release_pending = release_pending;





}