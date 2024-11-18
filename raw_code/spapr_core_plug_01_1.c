void spapr_core_plug(HotplugHandler *hotplug_dev, DeviceState *dev,

                     Error **errp)

{

    sPAPRMachineClass *smc = SPAPR_MACHINE_GET_CLASS(OBJECT(hotplug_dev));

    sPAPRMachineState *spapr = SPAPR_MACHINE(OBJECT(hotplug_dev));

    sPAPRCPUCore *core = SPAPR_CPU_CORE(OBJECT(dev));

    CPUCore *cc = CPU_CORE(dev);

    CPUState *cs = CPU(core->threads);

    sPAPRDRConnector *drc;

    sPAPRDRConnectorClass *drck;

    Error *local_err = NULL;

    void *fdt = NULL;

    int fdt_offset = 0;

    int index = cc->core_id / smp_threads;

    int smt = kvmppc_smt_threads();



    g_assert(smc->dr_cpu_enabled);



    drc = spapr_dr_connector_by_id(SPAPR_DR_CONNECTOR_TYPE_CPU, index * smt);

    spapr->cores[index] = OBJECT(dev);



    g_assert(drc);



    /*

     * Setup CPU DT entries only for hotplugged CPUs. For boot time or

     * coldplugged CPUs DT entries are setup in spapr_finalize_fdt().

     */

    if (dev->hotplugged) {

        fdt = spapr_populate_hotplug_cpu_dt(cs, &fdt_offset, spapr);

    }



    drck = SPAPR_DR_CONNECTOR_GET_CLASS(drc);

    drck->attach(drc, dev, fdt, fdt_offset, !dev->hotplugged, &local_err);

    if (local_err) {

        g_free(fdt);

        spapr->cores[index] = NULL;

        error_propagate(errp, local_err);

        return;

    }



    if (dev->hotplugged) {

        /*

         * Send hotplug notification interrupt to the guest only in case

         * of hotplugged CPUs.

         */

        spapr_hotplug_req_add_by_index(drc);

    } else {

        /*

         * Set the right DRC states for cold plugged CPU.

         */

        drck->set_allocation_state(drc, SPAPR_DR_ALLOCATION_STATE_USABLE);

        drck->set_isolation_state(drc, SPAPR_DR_ISOLATION_STATE_UNISOLATED);

    }

}
