static void spapr_dt_rtas(sPAPRMachineState *spapr, void *fdt)

{

    int rtas;

    GString *hypertas = g_string_sized_new(256);

    GString *qemu_hypertas = g_string_sized_new(256);

    uint32_t refpoints[] = { cpu_to_be32(0x4), cpu_to_be32(0x4) };

    uint64_t max_hotplug_addr = spapr->hotplug_memory.base +

        memory_region_size(&spapr->hotplug_memory.mr);

    uint32_t lrdr_capacity[] = {

        cpu_to_be32(max_hotplug_addr >> 32),

        cpu_to_be32(max_hotplug_addr & 0xffffffff),

        0, cpu_to_be32(SPAPR_MEMORY_BLOCK_SIZE),

        cpu_to_be32(max_cpus / smp_threads),

    };



    _FDT(rtas = fdt_add_subnode(fdt, 0, "rtas"));



    /* hypertas */

    add_str(hypertas, "hcall-pft");

    add_str(hypertas, "hcall-term");

    add_str(hypertas, "hcall-dabr");

    add_str(hypertas, "hcall-interrupt");

    add_str(hypertas, "hcall-tce");

    add_str(hypertas, "hcall-vio");

    add_str(hypertas, "hcall-splpar");

    add_str(hypertas, "hcall-bulk");

    add_str(hypertas, "hcall-set-mode");

    add_str(hypertas, "hcall-sprg0");

    add_str(hypertas, "hcall-copy");

    add_str(hypertas, "hcall-debug");

    add_str(qemu_hypertas, "hcall-memop1");



    if (!kvm_enabled() || kvmppc_spapr_use_multitce()) {

        add_str(hypertas, "hcall-multi-tce");

    }



    if (spapr->resize_hpt != SPAPR_RESIZE_HPT_DISABLED) {

        add_str(hypertas, "hcall-hpt-resize");

    }



    _FDT(fdt_setprop(fdt, rtas, "ibm,hypertas-functions",

                     hypertas->str, hypertas->len));

    g_string_free(hypertas, TRUE);

    _FDT(fdt_setprop(fdt, rtas, "qemu,hypertas-functions",

                     qemu_hypertas->str, qemu_hypertas->len));

    g_string_free(qemu_hypertas, TRUE);



    _FDT(fdt_setprop(fdt, rtas, "ibm,associativity-reference-points",

                     refpoints, sizeof(refpoints)));



    _FDT(fdt_setprop_cell(fdt, rtas, "rtas-error-log-max",

                          RTAS_ERROR_LOG_MAX));

    _FDT(fdt_setprop_cell(fdt, rtas, "rtas-event-scan-rate",

                          RTAS_EVENT_SCAN_RATE));



    if (msi_nonbroken) {

        _FDT(fdt_setprop(fdt, rtas, "ibm,change-msix-capable", NULL, 0));

    }



    /*

     * According to PAPR, rtas ibm,os-term does not guarantee a return

     * back to the guest cpu.

     *

     * While an additional ibm,extended-os-term property indicates

     * that rtas call return will always occur. Set this property.

     */

    _FDT(fdt_setprop(fdt, rtas, "ibm,extended-os-term", NULL, 0));



    _FDT(fdt_setprop(fdt, rtas, "ibm,lrdr-capacity",

                     lrdr_capacity, sizeof(lrdr_capacity)));



    spapr_dt_rtas_tokens(fdt, rtas);

}
