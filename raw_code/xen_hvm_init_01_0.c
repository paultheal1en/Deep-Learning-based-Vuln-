int xen_hvm_init(void)

{

    int i, rc;

    unsigned long ioreq_pfn;

    XenIOState *state;



    state = g_malloc0(sizeof (XenIOState));



    state->xce_handle = xen_xc_evtchn_open(NULL, 0);

    if (state->xce_handle == XC_HANDLER_INITIAL_VALUE) {

        perror("xen: event channel open");

        return -errno;

    }



    state->xenstore = xs_daemon_open();

    if (state->xenstore == NULL) {

        perror("xen: xenstore open");

        return -errno;

    }



    state->exit.notify = xen_exit_notifier;

    qemu_add_exit_notifier(&state->exit);



    state->suspend.notify = xen_suspend_notifier;

    qemu_register_suspend_notifier(&state->suspend);



    xc_get_hvm_param(xen_xc, xen_domid, HVM_PARAM_IOREQ_PFN, &ioreq_pfn);

    DPRINTF("shared page at pfn %lx\n", ioreq_pfn);

    state->shared_page = xc_map_foreign_range(xen_xc, xen_domid, XC_PAGE_SIZE,

                                              PROT_READ|PROT_WRITE, ioreq_pfn);

    if (state->shared_page == NULL) {

        hw_error("map shared IO page returned error %d handle=" XC_INTERFACE_FMT,

                 errno, xen_xc);

    }



    xc_get_hvm_param(xen_xc, xen_domid, HVM_PARAM_BUFIOREQ_PFN, &ioreq_pfn);

    DPRINTF("buffered io page at pfn %lx\n", ioreq_pfn);

    state->buffered_io_page = xc_map_foreign_range(xen_xc, xen_domid, XC_PAGE_SIZE,

                                                   PROT_READ|PROT_WRITE, ioreq_pfn);

    if (state->buffered_io_page == NULL) {

        hw_error("map buffered IO page returned error %d", errno);

    }



    state->ioreq_local_port = g_malloc0(smp_cpus * sizeof (evtchn_port_t));



    /* FIXME: how about if we overflow the page here? */

    for (i = 0; i < smp_cpus; i++) {

        rc = xc_evtchn_bind_interdomain(state->xce_handle, xen_domid,

                                        xen_vcpu_eport(state->shared_page, i));

        if (rc == -1) {

            fprintf(stderr, "bind interdomain ioctl error %d\n", errno);

            return -1;

        }

        state->ioreq_local_port[i] = rc;

    }



    /* Init RAM management */

    xen_map_cache_init();

    xen_ram_init(ram_size);



    qemu_add_vm_change_state_handler(xen_hvm_change_state_handler, state);



    state->memory_listener = xen_memory_listener;

    QLIST_INIT(&state->physmap);

    memory_listener_register(&state->memory_listener);

    state->log_for_dirtybit = NULL;



    /* Initialize backend core & drivers */

    if (xen_be_init() != 0) {

        fprintf(stderr, "%s: xen backend core setup failed\n", __FUNCTION__);

        exit(1);

    }

    xen_be_register("console", &xen_console_ops);

    xen_be_register("vkbd", &xen_kbdmouse_ops);

    xen_be_register("qdisk", &xen_blkdev_ops);

    xen_read_physmap(state);



    return 0;

}
