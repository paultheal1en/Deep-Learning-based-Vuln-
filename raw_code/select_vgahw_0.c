static void select_vgahw (const char *p)

{

    const char *opts;



    assert(vga_interface_type == VGA_NONE);

    if (strstart(p, "std", &opts)) {

        if (vga_available()) {

            vga_interface_type = VGA_STD;

        } else {

            error_report("standard VGA not available");

            exit(1);

        }

    } else if (strstart(p, "cirrus", &opts)) {

        if (cirrus_vga_available()) {

            vga_interface_type = VGA_CIRRUS;

        } else {

            error_report("Cirrus VGA not available");

            exit(1);

        }

    } else if (strstart(p, "vmware", &opts)) {

        if (vmware_vga_available()) {

            vga_interface_type = VGA_VMWARE;

        } else {

            error_report("VMWare SVGA not available");

            exit(1);

        }

    } else if (strstart(p, "virtio", &opts)) {

        if (virtio_vga_available()) {

            vga_interface_type = VGA_VIRTIO;

        } else {

            error_report("Virtio VGA not available");

            exit(1);

        }

    } else if (strstart(p, "xenfb", &opts)) {

        vga_interface_type = VGA_XENFB;

    } else if (strstart(p, "qxl", &opts)) {

        if (qxl_vga_available()) {

            vga_interface_type = VGA_QXL;

        } else {

            error_report("QXL VGA not available");

            exit(1);

        }

    } else if (strstart(p, "tcx", &opts)) {

        if (tcx_vga_available()) {

            vga_interface_type = VGA_TCX;

        } else {

            error_report("TCX framebuffer not available");

            exit(1);

        }

    } else if (strstart(p, "cg3", &opts)) {

        if (cg3_vga_available()) {

            vga_interface_type = VGA_CG3;

        } else {

            error_report("CG3 framebuffer not available");

            exit(1);

        }

    } else if (!strstart(p, "none", &opts)) {

    invalid_vga:

        error_report("unknown vga type: %s", p);

        exit(1);

    }

    while (*opts) {

        const char *nextopt;



        if (strstart(opts, ",retrace=", &nextopt)) {

            opts = nextopt;

            if (strstart(opts, "dumb", &nextopt))

                vga_retrace_method = VGA_RETRACE_DUMB;

            else if (strstart(opts, "precise", &nextopt))

                vga_retrace_method = VGA_RETRACE_PRECISE;

            else goto invalid_vga;

        } else goto invalid_vga;

        opts = nextopt;

    }

}
