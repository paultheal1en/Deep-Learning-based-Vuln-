static void *spapr_create_fdt_skel(hwaddr initrd_base,

                                   hwaddr initrd_size,

                                   hwaddr kernel_size,

                                   bool little_endian,

                                   const char *kernel_cmdline,

                                   uint32_t epow_irq)

{

    void *fdt;

    uint32_t start_prop = cpu_to_be32(initrd_base);

    uint32_t end_prop = cpu_to_be32(initrd_base + initrd_size);

    GString *hypertas = g_string_sized_new(256);

    GString *qemu_hypertas = g_string_sized_new(256);

    uint32_t refpoints[] = {cpu_to_be32(0x4), cpu_to_be32(0x4)};

    uint32_t interrupt_server_ranges_prop[] = {0, cpu_to_be32(max_cpus)};

    unsigned char vec5[] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x80};

    char *buf;



    add_str(hypertas, "hcall-pft");

    add_str(hypertas, "hcall-term");

    add_str(hypertas, "hcall-dabr");

    add_str(hypertas, "hcall-interrupt");

    add_str(hypertas, "hcall-tce");

    add_str(hypertas, "hcall-vio");

    add_str(hypertas, "hcall-splpar");

    add_str(hypertas, "hcall-bulk");

    add_str(hypertas, "hcall-set-mode");

    add_str(qemu_hypertas, "hcall-memop1");



    fdt = g_malloc0(FDT_MAX_SIZE);

    _FDT((fdt_create(fdt, FDT_MAX_SIZE)));



    if (kernel_size) {

        _FDT((fdt_add_reservemap_entry(fdt, KERNEL_LOAD_ADDR, kernel_size)));

    }

    if (initrd_size) {

        _FDT((fdt_add_reservemap_entry(fdt, initrd_base, initrd_size)));

    }

    _FDT((fdt_finish_reservemap(fdt)));



    /* Root node */

    _FDT((fdt_begin_node(fdt, "")));

    _FDT((fdt_property_string(fdt, "device_type", "chrp")));

    _FDT((fdt_property_string(fdt, "model", "IBM pSeries (emulated by qemu)")));

    _FDT((fdt_property_string(fdt, "compatible", "qemu,pseries")));



    /*

     * Add info to guest to indentify which host is it being run on

     * and what is the uuid of the guest

     */

    if (kvmppc_get_host_model(&buf)) {

        _FDT((fdt_property_string(fdt, "host-model", buf)));

        g_free(buf);

    }

    if (kvmppc_get_host_serial(&buf)) {

        _FDT((fdt_property_string(fdt, "host-serial", buf)));

        g_free(buf);

    }



    buf = g_strdup_printf(UUID_FMT, qemu_uuid[0], qemu_uuid[1],

                          qemu_uuid[2], qemu_uuid[3], qemu_uuid[4],

                          qemu_uuid[5], qemu_uuid[6], qemu_uuid[7],

                          qemu_uuid[8], qemu_uuid[9], qemu_uuid[10],

                          qemu_uuid[11], qemu_uuid[12], qemu_uuid[13],

                          qemu_uuid[14], qemu_uuid[15]);



    _FDT((fdt_property_string(fdt, "vm,uuid", buf)));

    if (qemu_uuid_set) {

        _FDT((fdt_property_string(fdt, "system-id", buf)));

    }

    g_free(buf);



    if (qemu_get_vm_name()) {

        _FDT((fdt_property_string(fdt, "ibm,partition-name",

                                  qemu_get_vm_name())));

    }



    _FDT((fdt_property_cell(fdt, "#address-cells", 0x2)));

    _FDT((fdt_property_cell(fdt, "#size-cells", 0x2)));



    /* /chosen */

    _FDT((fdt_begin_node(fdt, "chosen")));



    /* Set Form1_affinity */

    _FDT((fdt_property(fdt, "ibm,architecture-vec-5", vec5, sizeof(vec5))));



    _FDT((fdt_property_string(fdt, "bootargs", kernel_cmdline)));

    _FDT((fdt_property(fdt, "linux,initrd-start",

                       &start_prop, sizeof(start_prop))));

    _FDT((fdt_property(fdt, "linux,initrd-end",

                       &end_prop, sizeof(end_prop))));

    if (kernel_size) {

        uint64_t kprop[2] = { cpu_to_be64(KERNEL_LOAD_ADDR),

                              cpu_to_be64(kernel_size) };



        _FDT((fdt_property(fdt, "qemu,boot-kernel", &kprop, sizeof(kprop))));

        if (little_endian) {

            _FDT((fdt_property(fdt, "qemu,boot-kernel-le", NULL, 0)));

        }

    }

    if (boot_menu) {

        _FDT((fdt_property_cell(fdt, "qemu,boot-menu", boot_menu)));

    }

    _FDT((fdt_property_cell(fdt, "qemu,graphic-width", graphic_width)));

    _FDT((fdt_property_cell(fdt, "qemu,graphic-height", graphic_height)));

    _FDT((fdt_property_cell(fdt, "qemu,graphic-depth", graphic_depth)));



    _FDT((fdt_end_node(fdt)));



    /* RTAS */

    _FDT((fdt_begin_node(fdt, "rtas")));



    if (!kvm_enabled() || kvmppc_spapr_use_multitce()) {

        add_str(hypertas, "hcall-multi-tce");

    }

    _FDT((fdt_property(fdt, "ibm,hypertas-functions", hypertas->str,

                       hypertas->len)));

    g_string_free(hypertas, TRUE);

    _FDT((fdt_property(fdt, "qemu,hypertas-functions", qemu_hypertas->str,

                       qemu_hypertas->len)));

    g_string_free(qemu_hypertas, TRUE);



    _FDT((fdt_property(fdt, "ibm,associativity-reference-points",

        refpoints, sizeof(refpoints))));



    _FDT((fdt_property_cell(fdt, "rtas-error-log-max", RTAS_ERROR_LOG_MAX)));

    _FDT((fdt_property_cell(fdt, "rtas-event-scan-rate",

                            RTAS_EVENT_SCAN_RATE)));



    if (msi_nonbroken) {

        _FDT((fdt_property(fdt, "ibm,change-msix-capable", NULL, 0)));

    }



    /*

     * According to PAPR, rtas ibm,os-term does not guarantee a return

     * back to the guest cpu.

     *

     * While an additional ibm,extended-os-term property indicates that

     * rtas call return will always occur. Set this property.

     */

    _FDT((fdt_property(fdt, "ibm,extended-os-term", NULL, 0)));



    _FDT((fdt_end_node(fdt)));



    /* interrupt controller */

    _FDT((fdt_begin_node(fdt, "interrupt-controller")));



    _FDT((fdt_property_string(fdt, "device_type",

                              "PowerPC-External-Interrupt-Presentation")));

    _FDT((fdt_property_string(fdt, "compatible", "IBM,ppc-xicp")));

    _FDT((fdt_property(fdt, "interrupt-controller", NULL, 0)));

    _FDT((fdt_property(fdt, "ibm,interrupt-server-ranges",

                       interrupt_server_ranges_prop,

                       sizeof(interrupt_server_ranges_prop))));

    _FDT((fdt_property_cell(fdt, "#interrupt-cells", 2)));

    _FDT((fdt_property_cell(fdt, "linux,phandle", PHANDLE_XICP)));

    _FDT((fdt_property_cell(fdt, "phandle", PHANDLE_XICP)));



    _FDT((fdt_end_node(fdt)));



    /* vdevice */

    _FDT((fdt_begin_node(fdt, "vdevice")));



    _FDT((fdt_property_string(fdt, "device_type", "vdevice")));

    _FDT((fdt_property_string(fdt, "compatible", "IBM,vdevice")));

    _FDT((fdt_property_cell(fdt, "#address-cells", 0x1)));

    _FDT((fdt_property_cell(fdt, "#size-cells", 0x0)));

    _FDT((fdt_property_cell(fdt, "#interrupt-cells", 0x2)));

    _FDT((fdt_property(fdt, "interrupt-controller", NULL, 0)));



    _FDT((fdt_end_node(fdt)));



    /* event-sources */

    spapr_events_fdt_skel(fdt, epow_irq);



    /* /hypervisor node */

    if (kvm_enabled()) {

        uint8_t hypercall[16];



        /* indicate KVM hypercall interface */

        _FDT((fdt_begin_node(fdt, "hypervisor")));

        _FDT((fdt_property_string(fdt, "compatible", "linux,kvm")));

        if (kvmppc_has_cap_fixup_hcalls()) {

            /*

             * Older KVM versions with older guest kernels were broken with the

             * magic page, don't allow the guest to map it.

             */

            kvmppc_get_hypercall(first_cpu->env_ptr, hypercall,

                                 sizeof(hypercall));

            _FDT((fdt_property(fdt, "hcall-instructions", hypercall,

                              sizeof(hypercall))));

        }

        _FDT((fdt_end_node(fdt)));

    }



    _FDT((fdt_end_node(fdt))); /* close root node */

    _FDT((fdt_finish(fdt)));



    return fdt;

}
