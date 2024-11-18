static void xen_init_pv(MachineState *machine)

{

    DriveInfo *dinfo;

    int i;



    /* Initialize backend core & drivers */

    if (xen_be_init() != 0) {

        fprintf(stderr, "%s: xen backend core setup failed\n", __FUNCTION__);

        exit(1);

    }



    switch (xen_mode) {

    case XEN_ATTACH:

        /* nothing to do, xend handles everything */

        break;

#ifdef CONFIG_XEN_PV_DOMAIN_BUILD

    case XEN_CREATE: {

        const char *kernel_filename = machine->kernel_filename;

        const char *kernel_cmdline = machine->kernel_cmdline;

        const char *initrd_filename = machine->initrd_filename;

        if (xen_domain_build_pv(kernel_filename, initrd_filename,

                                kernel_cmdline) < 0) {

            fprintf(stderr, "xen pv domain creation failed\n");

            exit(1);

        }

        break;

    }

#endif

    case XEN_EMULATE:

        fprintf(stderr, "xen emulation not implemented (yet)\n");

        exit(1);

        break;

    default:

        fprintf(stderr, "unhandled xen_mode %d\n", xen_mode);

        exit(1);

        break;

    }



    xen_be_register_common();

    xen_be_register("vfb", &xen_framebuffer_ops);

    xen_be_register("qnic", &xen_netdev_ops);



    /* configure framebuffer */

    if (xenfb_enabled) {

        xen_config_dev_vfb(0, "vnc");

        xen_config_dev_vkbd(0);

    }



    /* configure disks */

    for (i = 0; i < 16; i++) {

        dinfo = drive_get(IF_XEN, 0, i);

        if (!dinfo)

            continue;

        xen_config_dev_blk(dinfo);

    }



    /* configure nics */

    for (i = 0; i < nb_nics; i++) {

        if (!nd_table[i].model || 0 != strcmp(nd_table[i].model, "xen"))

            continue;

        xen_config_dev_nic(nd_table + i);

    }



    /* config cleanup hook */

    atexit(xen_config_cleanup);



    /* setup framebuffer */

    xen_init_display(xen_domid);

}
