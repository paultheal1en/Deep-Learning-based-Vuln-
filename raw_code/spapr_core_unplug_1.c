void spapr_core_unplug(HotplugHandler *hotplug_dev, DeviceState *dev,

                       Error **errp)

{

    sPAPRCPUCore *core = SPAPR_CPU_CORE(OBJECT(dev));

    PowerPCCPU *cpu = POWERPC_CPU(core->threads);

    int id = ppc_get_vcpu_dt_id(cpu);

    sPAPRDRConnector *drc =

        spapr_dr_connector_by_id(SPAPR_DR_CONNECTOR_TYPE_CPU, id);

    sPAPRDRConnectorClass *drck;

    Error *local_err = NULL;



    g_assert(drc);



    drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

    drck->detach(drc, dev, spapr_core_release, NULL, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    spapr_hotplug_req_remove_by_index(drc);

}
