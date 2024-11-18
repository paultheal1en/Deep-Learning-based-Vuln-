static void *spapr_build_fdt(sPAPRMachineState *spapr,

                             hwaddr rtas_addr,

                             hwaddr rtas_size)

{

    MachineState *machine = MACHINE(qdev_get_machine());

    MachineClass *mc = MACHINE_GET_CLASS(machine);

    sPAPRMachineClass *smc = SPAPR_MACHINE_GET_CLASS(machine);

    int ret;

    void *fdt;

    sPAPRPHBState *phb;

    char *buf;



    fdt = g_malloc0(FDT_MAX_SIZE);

    _FDT((fdt_create_empty_tree(fdt, FDT_MAX_SIZE)));



    /* Root node */

    _FDT(fdt_setprop_string(fdt, 0, "device_type", "chrp"));

    _FDT(fdt_setprop_string(fdt, 0, "model", "IBM pSeries (emulated by qemu)"));

    _FDT(fdt_setprop_string(fdt, 0, "compatible", "qemu,pseries"));



    /*

     * Add info to guest to indentify which host is it being run on

     * and what is the uuid of the guest

     */

    if (kvmppc_get_host_model(&buf)) {

        _FDT(fdt_setprop_string(fdt, 0, "host-model", buf));

        g_free(buf);

    }

    if (kvmppc_get_host_serial(&buf)) {

        _FDT(fdt_setprop_string(fdt, 0, "host-serial", buf));

        g_free(buf);

    }



    buf = qemu_uuid_unparse_strdup(&qemu_uuid);



    _FDT(fdt_setprop_string(fdt, 0, "vm,uuid", buf));

    if (qemu_uuid_set) {

        _FDT(fdt_setprop_string(fdt, 0, "system-id", buf));

    }

    g_free(buf);



    if (qemu_get_vm_name()) {

        _FDT(fdt_setprop_string(fdt, 0, "ibm,partition-name",

                                qemu_get_vm_name()));

    }



    _FDT(fdt_setprop_cell(fdt, 0, "#address-cells", 2));

    _FDT(fdt_setprop_cell(fdt, 0, "#size-cells", 2));



    /* /interrupt controller */

    spapr_dt_xics(spapr->xics, fdt, PHANDLE_XICP);



    ret = spapr_populate_memory(spapr, fdt);

    if (ret < 0) {

        error_report("couldn't setup memory nodes in fdt");

        exit(1);

    }



    /* /vdevice */

    spapr_dt_vdevice(spapr->vio_bus, fdt);



    if (object_resolve_path_type("", TYPE_SPAPR_RNG, NULL)) {

        ret = spapr_rng_populate_dt(fdt);

        if (ret < 0) {

            error_report("could not set up rng device in the fdt");

            exit(1);

        }

    }



    QLIST_FOREACH(phb, &spapr->phbs, list) {

        ret = spapr_populate_pci_dt(phb, PHANDLE_XICP, fdt);

        if (ret < 0) {

            error_report("couldn't setup PCI devices in fdt");

            exit(1);

        }

    }



    /* cpus */

    spapr_populate_cpus_dt_node(fdt, spapr);



    if (smc->dr_lmb_enabled) {

        _FDT(spapr_drc_populate_dt(fdt, 0, NULL, SPAPR_DR_CONNECTOR_TYPE_LMB));

    }



    if (mc->query_hotpluggable_cpus) {

        int offset = fdt_path_offset(fdt, "/cpus");

        ret = spapr_drc_populate_dt(fdt, offset, NULL,

                                    SPAPR_DR_CONNECTOR_TYPE_CPU);

        if (ret < 0) {

            error_report("Couldn't set up CPU DR device tree properties");

            exit(1);

        }

    }



    /* /event-sources */

    spapr_dt_events(fdt, spapr->check_exception_irq);



    /* /rtas */

    spapr_dt_rtas(spapr, fdt);



    /* /chosen */

    spapr_dt_chosen(spapr, fdt);



    /* /hypervisor */

    if (kvm_enabled()) {

        spapr_dt_hypervisor(spapr, fdt);

    }



    /* Build memory reserve map */

    if (spapr->kernel_size) {

        _FDT((fdt_add_mem_rsv(fdt, KERNEL_LOAD_ADDR, spapr->kernel_size)));

    }

    if (spapr->initrd_size) {

        _FDT((fdt_add_mem_rsv(fdt, spapr->initrd_base, spapr->initrd_size)));

    }



    /* ibm,client-architecture-support updates */

    ret = spapr_dt_cas_updates(spapr, fdt, spapr->ov5_cas);

    if (ret < 0) {

        error_report("couldn't setup CAS properties fdt");

        exit(1);

    }



    return fdt;

}
