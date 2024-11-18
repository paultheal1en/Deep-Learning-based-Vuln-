int main(int argc, char **argv, char **envp)
    int i;
    int snapshot, linux_boot;
    const char *initrd_filename;
    const char *kernel_filename, *kernel_cmdline;
    const char *boot_order = NULL;
    const char *boot_once = NULL;
    DisplayState *ds;
    int cyls, heads, secs, translation;
    QemuOpts *opts, *machine_opts;
    QemuOpts *hda_opts = NULL, *icount_opts = NULL, *accel_opts = NULL;
    QemuOptsList *olist;
    int optind;
    const char *optarg;
    const char *loadvm = NULL;
    MachineClass *machine_class;
    const char *cpu_model;
    const char *vga_model = NULL;
    const char *qtest_chrdev = NULL;
    const char *qtest_log = NULL;
    const char *pid_file = NULL;
    const char *incoming = NULL;
    bool defconfig = true;
    bool userconfig = true;
    bool nographic = false;
    DisplayType display_type = DT_DEFAULT;
    int display_remote = 0;
    const char *log_mask = NULL;
    const char *log_file = NULL;
    char *trace_file = NULL;
    ram_addr_t maxram_size;
    uint64_t ram_slots = 0;
    FILE *vmstate_dump_file = NULL;
    Error *main_loop_err = NULL;
    Error *err = NULL;
    bool list_data_dirs = false;
    typedef struct BlockdevOptions_queue {
        BlockdevOptions *bdo;
        Location loc;
        QSIMPLEQ_ENTRY(BlockdevOptions_queue) entry;
    } BlockdevOptions_queue;
    QSIMPLEQ_HEAD(, BlockdevOptions_queue) bdo_queue
        = QSIMPLEQ_HEAD_INITIALIZER(bdo_queue);
    module_call_init(MODULE_INIT_TRACE);
    qemu_init_cpu_list();
    qemu_init_cpu_loop();
    qemu_mutex_lock_iothread();
    atexit(qemu_run_exit_notifiers);
    error_set_progname(argv[0]);
    qemu_init_exec_dir(argv[0]);
    module_call_init(MODULE_INIT_QOM);
    monitor_init_qmp_commands();
    qemu_add_opts(&qemu_drive_opts);
    qemu_add_drive_opts(&qemu_legacy_drive_opts);
    qemu_add_drive_opts(&qemu_common_drive_opts);
    qemu_add_drive_opts(&qemu_drive_opts);
    qemu_add_drive_opts(&bdrv_runtime_opts);
    qemu_add_opts(&qemu_chardev_opts);
    qemu_add_opts(&qemu_device_opts);
    qemu_add_opts(&qemu_netdev_opts);
    qemu_add_opts(&qemu_net_opts);
    qemu_add_opts(&qemu_rtc_opts);
    qemu_add_opts(&qemu_global_opts);
    qemu_add_opts(&qemu_mon_opts);
    qemu_add_opts(&qemu_trace_opts);
    qemu_add_opts(&qemu_option_rom_opts);
    qemu_add_opts(&qemu_machine_opts);
    qemu_add_opts(&qemu_accel_opts);
    qemu_add_opts(&qemu_mem_opts);
    qemu_add_opts(&qemu_smp_opts);
    qemu_add_opts(&qemu_boot_opts);
    qemu_add_opts(&qemu_sandbox_opts);
    qemu_add_opts(&qemu_add_fd_opts);
    qemu_add_opts(&qemu_object_opts);
    qemu_add_opts(&qemu_tpmdev_opts);
    qemu_add_opts(&qemu_realtime_opts);
    qemu_add_opts(&qemu_msg_opts);
    qemu_add_opts(&qemu_name_opts);
    qemu_add_opts(&qemu_numa_opts);
    qemu_add_opts(&qemu_icount_opts);
    qemu_add_opts(&qemu_semihosting_config_opts);
    qemu_add_opts(&qemu_fw_cfg_opts);
    module_call_init(MODULE_INIT_OPTS);
    runstate_init();
    if (qcrypto_init(&err) < 0) {
        error_reportf_err(err, "cannot initialize crypto: ");
    rtc_clock = QEMU_CLOCK_HOST;
    QLIST_INIT (&vm_change_state_head);
    os_setup_early_signal_handling();
    cpu_model = NULL;
    snapshot = 0;
    cyls = heads = secs = 0;
    translation = BIOS_ATA_TRANSLATION_AUTO;
    nb_nics = 0;
    bdrv_init_with_whitelist();
    autostart = 1;
    /* first pass of option parsing */
    optind = 1;
    while (optind < argc) {
        if (argv[optind][0] != '-') {
            /* disk image */
            optind++;
        } else {
            const QEMUOption *popt;
            popt = lookup_opt(argc, argv, &optarg, &optind);
            switch (popt->index) {
            case QEMU_OPTION_nodefconfig:
                defconfig = false;
            case QEMU_OPTION_nouserconfig:
                userconfig = false;
    if (defconfig && userconfig) {
        if (qemu_read_default_config_file() < 0) {
    /* second pass of option parsing */
    optind = 1;
    for(;;) {
        if (optind >= argc)
        if (argv[optind][0] != '-') {
            hda_opts = drive_add(IF_DEFAULT, 0, argv[optind++], HD_OPTS);
        } else {
            const QEMUOption *popt;
            popt = lookup_opt(argc, argv, &optarg, &optind);
            if (!(popt->arch_mask & arch_type)) {
                error_report("Option not supported for this target");
            switch(popt->index) {
            case QEMU_OPTION_no_kvm_irqchip: {
                olist = qemu_find_opts("machine");
                qemu_opts_parse_noisily(olist, "kernel_irqchip=off", false);
            case QEMU_OPTION_cpu:
                /* hw initialization will check this */
                cpu_model = optarg;
            case QEMU_OPTION_hda:
                    char buf[256];
                    if (cyls == 0)
                        snprintf(buf, sizeof(buf), "%s", HD_OPTS);
                    else
                        snprintf(buf, sizeof(buf),
                                 "%s,cyls=%d,heads=%d,secs=%d%s",
                                 HD_OPTS , cyls, heads, secs,
                                 translation == BIOS_ATA_TRANSLATION_LBA ?
                                 ",trans=lba" :
                                 translation == BIOS_ATA_TRANSLATION_NONE ?
                                 ",trans=none" : "");
                    drive_add(IF_DEFAULT, 0, optarg, buf);
            case QEMU_OPTION_hdb:
            case QEMU_OPTION_hdc:
            case QEMU_OPTION_hdd:
                drive_add(IF_DEFAULT, popt->index - QEMU_OPTION_hda, optarg,
                          HD_OPTS);
            case QEMU_OPTION_drive:
                if (drive_def(optarg) == NULL) {
            case QEMU_OPTION_set:
                if (qemu_set_option(optarg) != 0)
            case QEMU_OPTION_global:
                if (qemu_global_option(optarg) != 0)
            case QEMU_OPTION_mtdblock:
                drive_add(IF_MTD, -1, optarg, MTD_OPTS);
            case QEMU_OPTION_sd:
                drive_add(IF_SD, -1, optarg, SD_OPTS);
            case QEMU_OPTION_pflash:
                drive_add(IF_PFLASH, -1, optarg, PFLASH_OPTS);
            case QEMU_OPTION_snapshot:
                snapshot = 1;
            case QEMU_OPTION_hdachs:
                    const char *p;
                    p = optarg;
                    cyls = strtol(p, (char **)&p, 0);
                    if (cyls < 1 || cyls > 16383)
                        goto chs_fail;
                    if (*p != ',')
                        goto chs_fail;
                    p++;
                    heads = strtol(p, (char **)&p, 0);
                    if (heads < 1 || heads > 16)
                        goto chs_fail;
                    if (*p != ',')
                        goto chs_fail;
                    p++;
                    secs = strtol(p, (char **)&p, 0);
                    if (secs < 1 || secs > 63)
                        goto chs_fail;
                    if (*p == ',') {
                        p++;
                        if (!strcmp(p, "large")) {
                            translation = BIOS_ATA_TRANSLATION_LARGE;
                        } else if (!strcmp(p, "rechs")) {
                            translation = BIOS_ATA_TRANSLATION_RECHS;
                        } else if (!strcmp(p, "none")) {
                            translation = BIOS_ATA_TRANSLATION_NONE;
                        } else if (!strcmp(p, "lba")) {
                            translation = BIOS_ATA_TRANSLATION_LBA;
                        } else if (!strcmp(p, "auto")) {
                            translation = BIOS_ATA_TRANSLATION_AUTO;
                        } else {
                            goto chs_fail;
                    } else if (*p != '\0') {
                    chs_fail:
                        error_report("invalid physical CHS format");
                    if (hda_opts != NULL) {
                        qemu_opt_set_number(hda_opts, "cyls", cyls,
                                            &error_abort);
                        qemu_opt_set_number(hda_opts, "heads", heads,
                                            &error_abort);
                        qemu_opt_set_number(hda_opts, "secs", secs,
                                            &error_abort);
                        if (translation == BIOS_ATA_TRANSLATION_LARGE) {
                            qemu_opt_set(hda_opts, "trans", "large",
                                         &error_abort);
                        } else if (translation == BIOS_ATA_TRANSLATION_RECHS) {
                            qemu_opt_set(hda_opts, "trans", "rechs",
                                         &error_abort);
                        } else if (translation == BIOS_ATA_TRANSLATION_LBA) {
                            qemu_opt_set(hda_opts, "trans", "lba",
                                         &error_abort);
                        } else if (translation == BIOS_ATA_TRANSLATION_NONE) {
                            qemu_opt_set(hda_opts, "trans", "none",
                                         &error_abort);
            case QEMU_OPTION_numa:
                opts = qemu_opts_parse_noisily(qemu_find_opts("numa"),
                                               optarg, true);
                if (!opts) {
            case QEMU_OPTION_display:
                display_type = select_display(optarg);
            case QEMU_OPTION_nographic:
                olist = qemu_find_opts("machine");
                qemu_opts_parse_noisily(olist, "graphics=off", false);
                nographic = true;
                display_type = DT_NONE;
            case QEMU_OPTION_curses:
#ifdef CONFIG_CURSES
                display_type = DT_CURSES;
#else
                error_report("curses support is disabled");
#endif
            case QEMU_OPTION_portrait:
                graphic_rotate = 90;
            case QEMU_OPTION_rotate:
                graphic_rotate = strtol(optarg, (char **) &optarg, 10);
                if (graphic_rotate != 0 && graphic_rotate != 90 &&
                    graphic_rotate != 180 && graphic_rotate != 270) {
                    error_report("only 90, 180, 270 deg rotation is available");
            case QEMU_OPTION_kernel:
                qemu_opts_set(qemu_find_opts("machine"), 0, "kernel", optarg,
                              &error_abort);
            case QEMU_OPTION_initrd:
                qemu_opts_set(qemu_find_opts("machine"), 0, "initrd", optarg,
                              &error_abort);
            case QEMU_OPTION_append:
                qemu_opts_set(qemu_find_opts("machine"), 0, "append", optarg,
                              &error_abort);
            case QEMU_OPTION_dtb:
                qemu_opts_set(qemu_find_opts("machine"), 0, "dtb", optarg,
                              &error_abort);
            case QEMU_OPTION_cdrom:
                drive_add(IF_DEFAULT, 2, optarg, CDROM_OPTS);
            case QEMU_OPTION_boot:
                opts = qemu_opts_parse_noisily(qemu_find_opts("boot-opts"),
                                               optarg, true);
                if (!opts) {
            case QEMU_OPTION_fda:
            case QEMU_OPTION_fdb:
                drive_add(IF_FLOPPY, popt->index - QEMU_OPTION_fda,
                          optarg, FD_OPTS);
            case QEMU_OPTION_no_fd_bootchk:
                fd_bootchk = 0;
            case QEMU_OPTION_netdev:
                default_net = 0;
                if (net_client_parse(qemu_find_opts("netdev"), optarg) == -1) {
            case QEMU_OPTION_net:
                default_net = 0;
                if (net_client_parse(qemu_find_opts("net"), optarg) == -1) {
#ifdef CONFIG_LIBISCSI
            case QEMU_OPTION_iscsi:
                opts = qemu_opts_parse_noisily(qemu_find_opts("iscsi"),
                                               optarg, false);
                if (!opts) {
#endif
#ifdef CONFIG_SLIRP
            case QEMU_OPTION_tftp:
                error_report("The -tftp option is deprecated. "
                             "Please use '-netdev user,tftp=...' instead.");
                legacy_tftp_prefix = optarg;
            case QEMU_OPTION_bootp:
                error_report("The -bootp option is deprecated. "
                             "Please use '-netdev user,bootfile=...' instead.");
                legacy_bootp_filename = optarg;
            case QEMU_OPTION_redir:
                error_report("The -redir option is deprecated. "
                             "Please use '-netdev user,hostfwd=...' instead.");
                if (net_slirp_redir(optarg) < 0)
#endif
            case QEMU_OPTION_bt:
                add_device_config(DEV_BT, optarg);
            case QEMU_OPTION_audio_help:
                AUD_help ();
                exit (0);
            case QEMU_OPTION_soundhw:
                select_soundhw (optarg);
            case QEMU_OPTION_h:
                help(0);
            case QEMU_OPTION_version:
                version();
                exit(0);
            case QEMU_OPTION_m:
                opts = qemu_opts_parse_noisily(qemu_find_opts("memory"),
                                               optarg, true);
                if (!opts) {
                    exit(EXIT_FAILURE);
#ifdef CONFIG_TPM
            case QEMU_OPTION_tpmdev:
                if (tpm_config_parse(qemu_find_opts("tpmdev"), optarg) < 0) {
#endif
            case QEMU_OPTION_mempath:
                mem_path = optarg;
            case QEMU_OPTION_mem_prealloc:
                mem_prealloc = 1;
            case QEMU_OPTION_d:
                log_mask = optarg;
            case QEMU_OPTION_D:
                log_file = optarg;
            case QEMU_OPTION_DFILTER:
                qemu_set_dfilter_ranges(optarg, &error_fatal);
            case QEMU_OPTION_s:
                add_device_config(DEV_GDB, "tcp::" DEFAULT_GDBSTUB_PORT);
            case QEMU_OPTION_gdb:
                add_device_config(DEV_GDB, optarg);
            case QEMU_OPTION_L:
                if (is_help_option(optarg)) {
                    list_data_dirs = true;
                } else if (data_dir_idx < ARRAY_SIZE(data_dir)) {
                    data_dir[data_dir_idx++] = optarg;
            case QEMU_OPTION_bios:
                qemu_opts_set(qemu_find_opts("machine"), 0, "firmware", optarg,
                              &error_abort);
            case QEMU_OPTION_singlestep:
                singlestep = 1;
            case QEMU_OPTION_S:
                autostart = 0;
            case QEMU_OPTION_k:
                keyboard_layout = optarg;
            case QEMU_OPTION_localtime:
                rtc_utc = 0;
            case QEMU_OPTION_vga:
                vga_model = optarg;
                default_vga = 0;
            case QEMU_OPTION_g:
                    const char *p;
                    int w, h, depth;
                    p = optarg;
                    w = strtol(p, (char **)&p, 10);
                    if (w <= 0) {
                    graphic_error:
                        error_report("invalid resolution or depth");
                    if (*p != 'x')
                        goto graphic_error;
                    p++;
                    h = strtol(p, (char **)&p, 10);
                    if (h <= 0)
                        goto graphic_error;
                    if (*p == 'x') {
                        p++;
                        depth = strtol(p, (char **)&p, 10);
                        if (depth != 8 && depth != 15 && depth != 16 &&
                            depth != 24 && depth != 32)
                            goto graphic_error;
                    } else if (*p == '\0') {
                        depth = graphic_depth;
                    } else {
                        goto graphic_error;
                    graphic_width = w;
                    graphic_height = h;
                    graphic_depth = depth;
            case QEMU_OPTION_echr:
                    char *r;
                    term_escape_char = strtol(optarg, &r, 0);
                    if (r == optarg)
                        printf("Bad argument to echr\n");
            case QEMU_OPTION_monitor:
                default_monitor = 0;
                if (strncmp(optarg, "none", 4)) {
                    monitor_parse(optarg, "readline", false);
            case QEMU_OPTION_qmp:
                monitor_parse(optarg, "control", false);
                default_monitor = 0;
            case QEMU_OPTION_qmp_pretty:
                monitor_parse(optarg, "control", true);
                default_monitor = 0;
            case QEMU_OPTION_mon:
                opts = qemu_opts_parse_noisily(qemu_find_opts("mon"), optarg,
                                               true);
                if (!opts) {
                default_monitor = 0;
            case QEMU_OPTION_chardev:
                opts = qemu_opts_parse_noisily(qemu_find_opts("chardev"),
                                               optarg, true);
                if (!opts) {
            case QEMU_OPTION_fsdev:
                olist = qemu_find_opts("fsdev");
                if (!olist) {
                    error_report("fsdev support is disabled");
                opts = qemu_opts_parse_noisily(olist, optarg, true);
                if (!opts) {
            case QEMU_OPTION_virtfs: {
                QemuOpts *fsdev;
                QemuOpts *device;
                const char *writeout, *sock_fd, *socket;
                olist = qemu_find_opts("virtfs");
                if (!olist) {
                    error_report("virtfs support is disabled");
                opts = qemu_opts_parse_noisily(olist, optarg, true);
                if (!opts) {
                if (qemu_opt_get(opts, "fsdriver") == NULL ||
                    qemu_opt_get(opts, "mount_tag") == NULL) {
                    error_report("Usage: -virtfs fsdriver,mount_tag=tag");
                fsdev = qemu_opts_create(qemu_find_opts("fsdev"),
                                         qemu_opt_get(opts, "mount_tag"),
                                         1, NULL);
                if (!fsdev) {
                    error_report("duplicate fsdev id: %s",
                                 qemu_opt_get(opts, "mount_tag"));
                writeout = qemu_opt_get(opts, "writeout");
                if (writeout) {
#ifdef CONFIG_SYNC_FILE_RANGE
                    qemu_opt_set(fsdev, "writeout", writeout, &error_abort);
#else
                    error_report("writeout=immediate not supported "
                                 "on this platform");
#endif
                qemu_opt_set(fsdev, "fsdriver",
                             qemu_opt_get(opts, "fsdriver"), &error_abort);
                qemu_opt_set(fsdev, "path", qemu_opt_get(opts, "path"),
                             &error_abort);
                qemu_opt_set(fsdev, "security_model",
                             qemu_opt_get(opts, "security_model"),
                             &error_abort);
                socket = qemu_opt_get(opts, "socket");
                if (socket) {
                    qemu_opt_set(fsdev, "socket", socket, &error_abort);
                sock_fd = qemu_opt_get(opts, "sock_fd");
                if (sock_fd) {
                    qemu_opt_set(fsdev, "sock_fd", sock_fd, &error_abort);
                qemu_opt_set_bool(fsdev, "readonly",
                                  qemu_opt_get_bool(opts, "readonly", 0),
                                  &error_abort);
                device = qemu_opts_create(qemu_find_opts("device"), NULL, 0,
                                          &error_abort);
                qemu_opt_set(device, "driver", "virtio-9p-pci", &error_abort);
                qemu_opt_set(device, "fsdev",
                             qemu_opt_get(opts, "mount_tag"), &error_abort);
                qemu_opt_set(device, "mount_tag",
                             qemu_opt_get(opts, "mount_tag"), &error_abort);
            case QEMU_OPTION_virtfs_synth: {
                QemuOpts *fsdev;
                QemuOpts *device;
                fsdev = qemu_opts_create(qemu_find_opts("fsdev"), "v_synth",
                                         1, NULL);
                if (!fsdev) {
                    error_report("duplicate option: %s", "virtfs_synth");
                qemu_opt_set(fsdev, "fsdriver", "synth", &error_abort);
                device = qemu_opts_create(qemu_find_opts("device"), NULL, 0,
                                          &error_abort);
                qemu_opt_set(device, "driver", "virtio-9p-pci", &error_abort);
                qemu_opt_set(device, "fsdev", "v_synth", &error_abort);
                qemu_opt_set(device, "mount_tag", "v_synth", &error_abort);
            case QEMU_OPTION_serial:
                add_device_config(DEV_SERIAL, optarg);
                default_serial = 0;
                if (strncmp(optarg, "mon:", 4) == 0) {
                    default_monitor = 0;
            case QEMU_OPTION_watchdog:
                if (watchdog) {
                    error_report("only one watchdog option may be given");
                    return 1;
                watchdog = optarg;
            case QEMU_OPTION_watchdog_action:
                if (select_watchdog_action(optarg) == -1) {
                    error_report("unknown -watchdog-action parameter");
            case QEMU_OPTION_virtiocon:
                add_device_config(DEV_VIRTCON, optarg);
                default_virtcon = 0;
                if (strncmp(optarg, "mon:", 4) == 0) {
                    default_monitor = 0;
            case QEMU_OPTION_parallel:
                add_device_config(DEV_PARALLEL, optarg);
                default_parallel = 0;
                if (strncmp(optarg, "mon:", 4) == 0) {
                    default_monitor = 0;
            case QEMU_OPTION_debugcon:
                add_device_config(DEV_DEBUGCON, optarg);
            case QEMU_OPTION_loadvm:
                loadvm = optarg;
            case QEMU_OPTION_full_screen:
                full_screen = 1;
            case QEMU_OPTION_no_frame:
                no_frame = 1;
            case QEMU_OPTION_alt_grab:
                alt_grab = 1;
            case QEMU_OPTION_ctrl_grab:
                ctrl_grab = 1;
            case QEMU_OPTION_no_quit:
                no_quit = 1;
            case QEMU_OPTION_sdl:
#ifdef CONFIG_SDL
                display_type = DT_SDL;
#else
                error_report("SDL support is disabled");
#endif
            case QEMU_OPTION_pidfile:
                pid_file = optarg;
            case QEMU_OPTION_win2k_hack:
                win2k_install_hack = 1;
            case QEMU_OPTION_rtc_td_hack: {
                static GlobalProperty slew_lost_ticks = {
                    .driver   = "mc146818rtc",
                    .property = "lost_tick_policy",
                    .value    = "slew",
                };
                qdev_prop_register_global(&slew_lost_ticks);
            case QEMU_OPTION_acpitable:
                opts = qemu_opts_parse_noisily(qemu_find_opts("acpi"),
                                               optarg, true);
                if (!opts) {
                acpi_table_add(opts, &error_fatal);
            case QEMU_OPTION_smbios:
                opts = qemu_opts_parse_noisily(qemu_find_opts("smbios"),
                                               optarg, false);
                if (!opts) {
                smbios_entry_add(opts, &error_fatal);
            case QEMU_OPTION_fwcfg:
                opts = qemu_opts_parse_noisily(qemu_find_opts("fw_cfg"),
                                               optarg, true);
                if (opts == NULL) {
            case QEMU_OPTION_enable_kvm:
                olist = qemu_find_opts("machine");
                qemu_opts_parse_noisily(olist, "accel=kvm", false);
            case QEMU_OPTION_enable_hax:
                olist = qemu_find_opts("machine");
                qemu_opts_parse_noisily(olist, "accel=hax", false);
            case QEMU_OPTION_M:
            case QEMU_OPTION_machine:
                olist = qemu_find_opts("machine");
                opts = qemu_opts_parse_noisily(olist, optarg, true);
                if (!opts) {
             case QEMU_OPTION_no_kvm:
                olist = qemu_find_opts("machine");
                qemu_opts_parse_noisily(olist, "accel=tcg", false);
            case QEMU_OPTION_no_kvm_pit: {
                error_report("warning: ignoring deprecated option");
            case QEMU_OPTION_no_kvm_pit_reinjection: {
                static GlobalProperty kvm_pit_lost_tick_policy = {
                    .driver   = "kvm-pit",
                    .property = "lost_tick_policy",
                    .value    = "discard",
                };
                error_report("warning: deprecated, replaced by "
                             "-global kvm-pit.lost_tick_policy=discard");
                qdev_prop_register_global(&kvm_pit_lost_tick_policy);
            case QEMU_OPTION_accel:
                accel_opts = qemu_opts_parse_noisily(qemu_find_opts("accel"),
                                                     optarg, true);
                optarg = qemu_opt_get(accel_opts, "accel");
                olist = qemu_find_opts("machine");
                if (strcmp("kvm", optarg) == 0) {
                    qemu_opts_parse_noisily(olist, "accel=kvm", false);
                } else if (strcmp("xen", optarg) == 0) {
                    qemu_opts_parse_noisily(olist, "accel=xen", false);
                } else if (strcmp("tcg", optarg) == 0) {
                    qemu_opts_parse_noisily(olist, "accel=tcg", false);
                } else {
                    if (!is_help_option(optarg)) {
                        error_printf("Unknown accelerator: %s", optarg);
                    error_printf("Supported accelerators: kvm, xen, tcg\n");
            case QEMU_OPTION_usb:
                olist = qemu_find_opts("machine");
                qemu_opts_parse_noisily(olist, "usb=on", false);
            case QEMU_OPTION_usbdevice:
                olist = qemu_find_opts("machine");
                qemu_opts_parse_noisily(olist, "usb=on", false);
                add_device_config(DEV_USB, optarg);
            case QEMU_OPTION_device:
                if (!qemu_opts_parse_noisily(qemu_find_opts("device"),
                                             optarg, true)) {
            case QEMU_OPTION_smp:
                if (!qemu_opts_parse_noisily(qemu_find_opts("smp-opts"),
                                             optarg, true)) {
            case QEMU_OPTION_vnc:
                vnc_parse(optarg, &error_fatal);
            case QEMU_OPTION_no_acpi:
                acpi_enabled = 0;
            case QEMU_OPTION_no_hpet:
                no_hpet = 1;
            case QEMU_OPTION_balloon:
                if (balloon_parse(optarg) < 0) {
                    error_report("unknown -balloon argument %s", optarg);
            case QEMU_OPTION_no_reboot:
                no_reboot = 1;
            case QEMU_OPTION_no_shutdown:
                no_shutdown = 1;
            case QEMU_OPTION_show_cursor:
                cursor_hide = 0;
            case QEMU_OPTION_uuid:
                if (qemu_uuid_parse(optarg, &qemu_uuid) < 0) {
                    error_report("failed to parse UUID string: wrong format");
                qemu_uuid_set = true;
            case QEMU_OPTION_option_rom:
                if (nb_option_roms >= MAX_OPTION_ROMS) {
                    error_report("too many option ROMs");
                opts = qemu_opts_parse_noisily(qemu_find_opts("option-rom"),
                                               optarg, true);
                if (!opts) {
                option_rom[nb_option_roms].name = qemu_opt_get(opts, "romfile");
                option_rom[nb_option_roms].bootindex =
                    qemu_opt_get_number(opts, "bootindex", -1);
                if (!option_rom[nb_option_roms].name) {
                    error_report("Option ROM file is not specified");
                nb_option_roms++;
            case QEMU_OPTION_semihosting:
                semihosting.enabled = true;
                semihosting.target = SEMIHOSTING_TARGET_AUTO;
            case QEMU_OPTION_semihosting_config:
                semihosting.enabled = true;
                opts = qemu_opts_parse_noisily(qemu_find_opts("semihosting-config"),
                                               optarg, false);
                if (opts != NULL) {
                    semihosting.enabled = qemu_opt_get_bool(opts, "enable",
                                                            true);
                    const char *target = qemu_opt_get(opts, "target");
                    if (target != NULL) {
                        if (strcmp("native", target) == 0) {
                            semihosting.target = SEMIHOSTING_TARGET_NATIVE;
                        } else if (strcmp("gdb", target) == 0) {
                            semihosting.target = SEMIHOSTING_TARGET_GDB;
                        } else  if (strcmp("auto", target) == 0) {
                            semihosting.target = SEMIHOSTING_TARGET_AUTO;
                        } else {
                            error_report("unsupported semihosting-config %s",
                                         optarg);
                    } else {
                        semihosting.target = SEMIHOSTING_TARGET_AUTO;
                    /* Set semihosting argument count and vector */
                    qemu_opt_foreach(opts, add_semihosting_arg,
                                     &semihosting, NULL);
                } else {
                    error_report("unsupported semihosting-config %s", optarg);
            case QEMU_OPTION_tdf:
                error_report("warning: ignoring deprecated option");
            case QEMU_OPTION_name:
                opts = qemu_opts_parse_noisily(qemu_find_opts("name"),
                                               optarg, true);
                if (!opts) {
            case QEMU_OPTION_prom_env:
                if (nb_prom_envs >= MAX_PROM_ENVS) {
                    error_report("too many prom variables");
                prom_envs[nb_prom_envs] = optarg;
                nb_prom_envs++;
            case QEMU_OPTION_old_param:
                old_param = 1;
            case QEMU_OPTION_clock:
                /* Clock options no longer exist.  Keep this option for
                 * backward compatibility.
                 */
            case QEMU_OPTION_startdate:
                configure_rtc_date_offset(optarg, 1);
            case QEMU_OPTION_rtc:
                opts = qemu_opts_parse_noisily(qemu_find_opts("rtc"), optarg,
                                               false);
                if (!opts) {
                configure_rtc(opts);
            case QEMU_OPTION_tb_size:
                tcg_tb_size = strtol(optarg, NULL, 0);
                if (tcg_tb_size < 0) {
                    tcg_tb_size = 0;
            case QEMU_OPTION_icount:
                icount_opts = qemu_opts_parse_noisily(qemu_find_opts("icount"),
                                                      optarg, true);
                if (!icount_opts) {
            case QEMU_OPTION_incoming:
                if (!incoming) {
                    runstate_set(RUN_STATE_INMIGRATE);
                incoming = optarg;
            case QEMU_OPTION_only_migratable:
                only_migratable = 1;
            case QEMU_OPTION_nodefaults:
                has_defaults = 0;
            case QEMU_OPTION_xen_domid:
                if (!(xen_available())) {
                    error_report("Option not supported for this target");
                xen_domid = atoi(optarg);
            case QEMU_OPTION_xen_create:
                if (!(xen_available())) {
                    error_report("Option not supported for this target");
                xen_mode = XEN_CREATE;
            case QEMU_OPTION_xen_attach:
                if (!(xen_available())) {
                    error_report("Option not supported for this target");
                xen_mode = XEN_ATTACH;
            case QEMU_OPTION_trace:
                g_free(trace_file);
                trace_file = trace_opt_parse(optarg);
            case QEMU_OPTION_readconfig:
                    int ret = qemu_read_config_file(optarg);
                    if (ret < 0) {
                        error_report("read config %s: %s", optarg,
                                     strerror(-ret));
            case QEMU_OPTION_spice:
                olist = qemu_find_opts("spice");
                if (!olist) {
                    error_report("spice support is disabled");
                opts = qemu_opts_parse_noisily(olist, optarg, false);
                if (!opts) {
                display_remote++;
            case QEMU_OPTION_writeconfig:
                    FILE *fp;
                    if (strcmp(optarg, "-") == 0) {
                        fp = stdout;
                    } else {
                        fp = fopen(optarg, "w");
                        if (fp == NULL) {
                            error_report("open %s: %s", optarg,
                                         strerror(errno));
                    qemu_config_write(fp);
                    if (fp != stdout) {
                        fclose(fp);
            case QEMU_OPTION_qtest:
                qtest_chrdev = optarg;
            case QEMU_OPTION_qtest_log:
                qtest_log = optarg;
            case QEMU_OPTION_sandbox:
                opts = qemu_opts_parse_noisily(qemu_find_opts("sandbox"),
                                               optarg, true);
                if (!opts) {
            case QEMU_OPTION_add_fd:
#ifndef _WIN32
                opts = qemu_opts_parse_noisily(qemu_find_opts("add-fd"),
                                               optarg, false);
                if (!opts) {
#else
                error_report("File descriptor passing is disabled on this "
                             "platform");
#endif
            case QEMU_OPTION_object:
                opts = qemu_opts_parse_noisily(qemu_find_opts("object"),
                                               optarg, true);
                if (!opts) {
            case QEMU_OPTION_realtime:
                opts = qemu_opts_parse_noisily(qemu_find_opts("realtime"),
                                               optarg, false);
                if (!opts) {
                enable_mlock = qemu_opt_get_bool(opts, "mlock", true);
            case QEMU_OPTION_msg:
                opts = qemu_opts_parse_noisily(qemu_find_opts("msg"), optarg,
                                               false);
                if (!opts) {
                configure_msg(opts);
            case QEMU_OPTION_dump_vmstate:
                if (vmstate_dump_file) {
                    error_report("only one '-dump-vmstate' "
                                 "option may be given");
                vmstate_dump_file = fopen(optarg, "w");
                if (vmstate_dump_file == NULL) {
                    error_report("open %s: %s", optarg, strerror(errno));
            default:
                os_parse_cmd_args(popt->index, optarg);
    /*
     * Clear error location left behind by the loop.
     * Best done right after the loop.  Do not insert code here!
     */
    loc_set_none();
    replay_configure(icount_opts);
    qemu_tcg_configure(accel_opts, &error_fatal);
    machine_class = select_machine();
    set_memory_options(&ram_slots, &maxram_size, machine_class);
    os_daemonize();
    if (pid_file && qemu_create_pidfile(pid_file) != 0) {
        error_report("could not acquire pid file: %s", strerror(errno));
    if (qemu_init_main_loop(&main_loop_err)) {
        error_report_err(main_loop_err);
    if (qemu_opts_foreach(qemu_find_opts("sandbox"),
                          parse_sandbox, NULL, NULL)) {
    if (qemu_opts_foreach(qemu_find_opts("name"),
                          parse_name, NULL, NULL)) {
#ifndef _WIN32
    if (qemu_opts_foreach(qemu_find_opts("add-fd"),
                          parse_add_fd, NULL, NULL)) {
    if (qemu_opts_foreach(qemu_find_opts("add-fd"),
                          cleanup_add_fd, NULL, NULL)) {
#endif
    current_machine = MACHINE(object_new(object_class_get_name(
                          OBJECT_CLASS(machine_class))));
    if (machine_help_func(qemu_get_machine_opts(), current_machine)) {
        exit(0);
    object_property_add_child(object_get_root(), "machine",
                              OBJECT(current_machine), &error_abort);
    if (machine_class->minimum_page_bits) {
        if (!set_preferred_target_page_bits(machine_class->minimum_page_bits)) {
            /* This would be a board error: specifying a minimum smaller than
             * a target's compile-time fixed setting.
             */
            g_assert_not_reached();
    cpu_exec_init_all();
    if (machine_class->hw_version) {
        qemu_set_hw_version(machine_class->hw_version);
    if (cpu_model && is_help_option(cpu_model)) {
        list_cpus(stdout, &fprintf, cpu_model);
        exit(0);
    if (!trace_init_backends()) {
    trace_init_file(trace_file);
    /* Open the logfile at this point and set the log mask if necessary.
     */
    if (log_file) {
        qemu_set_log_filename(log_file, &error_fatal);
    if (log_mask) {
        int mask;
        mask = qemu_str_to_log_mask(log_mask);
        if (!mask) {
            qemu_print_log_usage(stdout);
        qemu_set_log(mask);
    } else {
        qemu_set_log(0);
    /* If no data_dir is specified then try to find it relative to the
       executable path.  */
    if (data_dir_idx < ARRAY_SIZE(data_dir)) {
        data_dir[data_dir_idx] = os_find_datadir();
        if (data_dir[data_dir_idx] != NULL) {
            data_dir_idx++;
    /* If all else fails use the install path specified when building. */
    if (data_dir_idx < ARRAY_SIZE(data_dir)) {
        data_dir[data_dir_idx++] = CONFIG_QEMU_DATADIR;
    /* -L help lists the data directories and exits. */
    if (list_data_dirs) {
        for (i = 0; i < data_dir_idx; i++) {
            printf("%s\n", data_dir[i]);
        exit(0);
    smp_parse(qemu_opts_find(qemu_find_opts("smp-opts"), NULL));
    machine_class->max_cpus = machine_class->max_cpus ?: 1; /* Default to UP */
    if (max_cpus > machine_class->max_cpus) {
        error_report("Number of SMP CPUs requested (%d) exceeds max CPUs "
                     "supported by machine '%s' (%d)", max_cpus,
                     machine_class->name, machine_class->max_cpus);
    /*
     * Get the default machine options from the machine if it is not already
     * specified either by the configuration file or by the command line.
     */
    if (machine_class->default_machine_opts) {
        qemu_opts_set_defaults(qemu_find_opts("machine"),
                               machine_class->default_machine_opts, 0);
    qemu_opts_foreach(qemu_find_opts("device"),
                      default_driver_check, NULL, NULL);
    qemu_opts_foreach(qemu_find_opts("global"),
                      default_driver_check, NULL, NULL);
    if (!vga_model && !default_vga) {
        vga_interface_type = VGA_DEVICE;
    if (!has_defaults || machine_class->no_serial) {
        default_serial = 0;
    if (!has_defaults || machine_class->no_parallel) {
        default_parallel = 0;
    if (!has_defaults || !machine_class->use_virtcon) {
        default_virtcon = 0;
    if (!has_defaults || !machine_class->use_sclp) {
        default_sclp = 0;
    if (!has_defaults || machine_class->no_floppy) {
        default_floppy = 0;
    if (!has_defaults || machine_class->no_cdrom) {
        default_cdrom = 0;
    if (!has_defaults || machine_class->no_sdcard) {
        default_sdcard = 0;
    if (!has_defaults) {
        default_monitor = 0;
        default_net = 0;
        default_vga = 0;
    if (is_daemonized()) {
        /* According to documentation and historically, -nographic redirects
         * serial port, parallel port and monitor to stdio, which does not work
         * with -daemonize.  We can redirect these to null instead, but since
         * -nographic is legacy, let's just error out.
         * We disallow -nographic only if all other ports are not redirected
         * explicitly, to not break existing legacy setups which uses
         * -nographic _and_ redirects all ports explicitly - this is valid
         * usage, -nographic is just a no-op in this case.
         */
        if (nographic
            && (default_parallel || default_serial
                || default_monitor || default_virtcon)) {
            error_report("-nographic cannot be used with -daemonize");
#ifdef CONFIG_CURSES
        if (display_type == DT_CURSES) {
            error_report("curses display cannot be used with -daemonize");
#endif
    if (nographic) {
        if (default_parallel)
            add_device_config(DEV_PARALLEL, "null");
        if (default_serial && default_monitor) {
            add_device_config(DEV_SERIAL, "mon:stdio");
        } else if (default_virtcon && default_monitor) {
            add_device_config(DEV_VIRTCON, "mon:stdio");
        } else if (default_sclp && default_monitor) {
            add_device_config(DEV_SCLP, "mon:stdio");
        } else {
            if (default_serial)
                add_device_config(DEV_SERIAL, "stdio");
            if (default_virtcon)
                add_device_config(DEV_VIRTCON, "stdio");
            if (default_sclp) {
                add_device_config(DEV_SCLP, "stdio");
            if (default_monitor)
                monitor_parse("stdio", "readline", false);
    } else {
        if (default_serial)
            add_device_config(DEV_SERIAL, "vc:80Cx24C");
        if (default_parallel)
            add_device_config(DEV_PARALLEL, "vc:80Cx24C");
        if (default_monitor)
            monitor_parse("vc:80Cx24C", "readline", false);
        if (default_virtcon)
            add_device_config(DEV_VIRTCON, "vc:80Cx24C");
        if (default_sclp) {
            add_device_config(DEV_SCLP, "vc:80Cx24C");
#if defined(CONFIG_VNC)
    if (!QTAILQ_EMPTY(&(qemu_find_opts("vnc")->head))) {
        display_remote++;
#endif
    if (display_type == DT_DEFAULT && !display_remote) {
#if defined(CONFIG_GTK)
        display_type = DT_GTK;
#elif defined(CONFIG_SDL)
        display_type = DT_SDL;
#elif defined(CONFIG_COCOA)
        display_type = DT_COCOA;
#elif defined(CONFIG_VNC)
        vnc_parse("localhost:0,to=99,id=default", &error_abort);
#else
        display_type = DT_NONE;
#endif
    if ((no_frame || alt_grab || ctrl_grab) && display_type != DT_SDL) {
        error_report("-no-frame, -alt-grab and -ctrl-grab are only valid "
                     "for SDL, ignoring option");
    if (no_quit && (display_type != DT_GTK && display_type != DT_SDL)) {
        error_report("-no-quit is only valid for GTK and SDL, "
                     "ignoring option");
    if (display_type == DT_GTK) {
        early_gtk_display_init(request_opengl);
    if (display_type == DT_SDL) {
        sdl_display_early_init(request_opengl);
    qemu_console_early_init();
    if (request_opengl == 1 && display_opengl == 0) {
#if defined(CONFIG_OPENGL)
        error_report("OpenGL is not supported by the display");
#else
        error_report("OpenGL support is disabled");
#endif
    page_size_init();
    socket_init();
    if (qemu_opts_foreach(qemu_find_opts("object"),
                          user_creatable_add_opts_foreach,
                          object_create_initial, NULL)) {
    if (qemu_opts_foreach(qemu_find_opts("chardev"),
                          chardev_init_func, NULL, NULL)) {
#ifdef CONFIG_VIRTFS
    if (qemu_opts_foreach(qemu_find_opts("fsdev"),
                          fsdev_init_func, NULL, NULL)) {
#endif
    if (qemu_opts_foreach(qemu_find_opts("device"),
                          device_help_func, NULL, NULL)) {
        exit(0);
    machine_opts = qemu_get_machine_opts();
    if (qemu_opt_foreach(machine_opts, machine_set_property, current_machine,
                         NULL)) {
        object_unref(OBJECT(current_machine));
    configure_accelerator(current_machine);
    if (qtest_chrdev) {
        qtest_init(qtest_chrdev, qtest_log, &error_fatal);
    machine_opts = qemu_get_machine_opts();
    kernel_filename = qemu_opt_get(machine_opts, "kernel");
    initrd_filename = qemu_opt_get(machine_opts, "initrd");
    kernel_cmdline = qemu_opt_get(machine_opts, "append");
    bios_name = qemu_opt_get(machine_opts, "firmware");
    opts = qemu_opts_find(qemu_find_opts("boot-opts"), NULL);
    if (opts) {
        boot_order = qemu_opt_get(opts, "order");
        if (boot_order) {
            validate_bootdevices(boot_order, &error_fatal);
        boot_once = qemu_opt_get(opts, "once");
        if (boot_once) {
            validate_bootdevices(boot_once, &error_fatal);
        boot_menu = qemu_opt_get_bool(opts, "menu", boot_menu);
        boot_strict = qemu_opt_get_bool(opts, "strict", false);
    if (!boot_order) {
        boot_order = machine_class->default_boot_order;
    if (!kernel_cmdline) {
        kernel_cmdline = "";
        current_machine->kernel_cmdline = (char *)kernel_cmdline;
    linux_boot = (kernel_filename != NULL);
    if (!linux_boot && *kernel_cmdline != '\0') {
        error_report("-append only allowed with -kernel option");
    if (!linux_boot && initrd_filename != NULL) {
        error_report("-initrd only allowed with -kernel option");
    if (semihosting_enabled() && !semihosting_get_argc() && kernel_filename) {
        /* fall back to the -kernel/-append */
        semihosting_arg_fallback(kernel_filename, kernel_cmdline);
    os_set_line_buffering();
    /* spice needs the timers to be initialized by this point */
    qemu_spice_init();
    cpu_ticks_init();
    if (icount_opts) {
        if (!tcg_enabled()) {
            error_report("-icount is not allowed with hardware virtualization");
        } else if (qemu_tcg_mttcg_enabled()) {
            error_report("-icount does not currently work with MTTCG");
        configure_icount(icount_opts, &error_abort);
        qemu_opts_del(icount_opts);
    if (default_net) {
        QemuOptsList *net = qemu_find_opts("net");
        qemu_opts_set(net, NULL, "type", "nic", &error_abort);
#ifdef CONFIG_SLIRP
        qemu_opts_set(net, NULL, "type", "user", &error_abort);
#endif
    colo_info_init();
    if (net_init_clients() < 0) {
    if (qemu_opts_foreach(qemu_find_opts("object"),
                          user_creatable_add_opts_foreach,
                          object_create_delayed, NULL)) {
#ifdef CONFIG_TPM
    if (tpm_init() < 0) {
#endif
    /* init the bluetooth world */
    if (foreach_device_config(DEV_BT, bt_parse))
    if (!xen_enabled()) {
        /* On 32-bit hosts, QEMU is limited by virtual address space */
        if (ram_size > (2047 << 20) && HOST_LONG_BITS == 32) {
            error_report("at most 2047 MB RAM can be simulated");
    blk_mig_init();
    ram_mig_init();
    /* If the currently selected machine wishes to override the units-per-bus
     * property of its default HBA interface type, do so now. */
    if (machine_class->units_per_default_bus) {
        override_max_devs(machine_class->block_default_type,
                          machine_class->units_per_default_bus);
    /* open the virtual block devices */
    while (!QSIMPLEQ_EMPTY(&bdo_queue)) {
        BlockdevOptions_queue *bdo = QSIMPLEQ_FIRST(&bdo_queue);
        QSIMPLEQ_REMOVE_HEAD(&bdo_queue, entry);
        loc_push_restore(&bdo->loc);
        qmp_blockdev_add(bdo->bdo, &error_fatal);
        loc_pop(&bdo->loc);
        qapi_free_BlockdevOptions(bdo->bdo);
        g_free(bdo);
    if (snapshot || replay_mode != REPLAY_MODE_NONE) {
        qemu_opts_foreach(qemu_find_opts("drive"), drive_enable_snapshot,
                          NULL, NULL);
    if (qemu_opts_foreach(qemu_find_opts("drive"), drive_init_func,
                          &machine_class->block_default_type, NULL)) {
    default_drive(default_cdrom, snapshot, machine_class->block_default_type, 2,
                  CDROM_OPTS);
    default_drive(default_floppy, snapshot, IF_FLOPPY, 0, FD_OPTS);
    default_drive(default_sdcard, snapshot, IF_SD, 0, SD_OPTS);
    parse_numa_opts(machine_class);
    if (qemu_opts_foreach(qemu_find_opts("mon"),
                          mon_init_func, NULL, NULL)) {
    if (foreach_device_config(DEV_SERIAL, serial_parse) < 0)
    if (foreach_device_config(DEV_PARALLEL, parallel_parse) < 0)
    if (foreach_device_config(DEV_VIRTCON, virtcon_parse) < 0)
    if (foreach_device_config(DEV_SCLP, sclp_parse) < 0) {
    if (foreach_device_config(DEV_DEBUGCON, debugcon_parse) < 0)
    /* If no default VGA is requested, the default is "none".  */
    if (default_vga) {
        if (machine_class->default_display) {
            vga_model = machine_class->default_display;
        } else if (vga_interface_available(VGA_CIRRUS)) {
            vga_model = "cirrus";
        } else if (vga_interface_available(VGA_STD)) {
            vga_model = "std";
    if (vga_model) {
        select_vgahw(vga_model);
    if (watchdog) {
        i = select_watchdog(watchdog);
        if (i > 0)
            exit (i == 1 ? 1 : 0);
    machine_register_compat_props(current_machine);
    qemu_opts_foreach(qemu_find_opts("global"),
                      global_init_func, NULL, NULL);
    /* This checkpoint is required by replay to separate prior clock
       reading from the other reads, because timer polling functions query
       clock values from the log. */
    replay_checkpoint(CHECKPOINT_INIT);
    qdev_machine_init();
    current_machine->ram_size = ram_size;
    current_machine->maxram_size = maxram_size;
    current_machine->ram_slots = ram_slots;
    current_machine->boot_order = boot_order;
    current_machine->cpu_model = cpu_model;
    machine_class->init(current_machine);
    realtime_init();
    audio_init();
    if (hax_enabled()) {
        hax_sync_vcpus();
    if (qemu_opts_foreach(qemu_find_opts("fw_cfg"),
                          parse_fw_cfg, fw_cfg_find(), NULL) != 0) {
    /* init USB devices */
    if (machine_usb(current_machine)) {
        if (foreach_device_config(DEV_USB, usb_parse) < 0)
    /* Check if IGD GFX passthrough. */
    igd_gfx_passthru();
    /* init generic devices */
    rom_set_order_override(FW_CFG_ORDER_OVERRIDE_DEVICE);
    if (qemu_opts_foreach(qemu_find_opts("device"),
                          device_init_func, NULL, NULL)) {
    cpu_synchronize_all_post_init();
    numa_post_machine_init();
    rom_reset_order_override();
    /*
     * Create frontends for -drive if=scsi leftovers.
     * Normally, frontends for -drive get created by machine
     * initialization for onboard SCSI HBAs.  However, we create a few
     * more ever since SCSI qdevification, but this is pretty much an
     * implementation accident, and deprecated.
     */
    scsi_legacy_handle_cmdline();
    /* Did we create any drives that we failed to create a device for? */
    drive_check_orphaned();
    /* Don't warn about the default network setup that you get if
     * no command line -net or -netdev options are specified. There
     * are two cases that we would otherwise complain about:
     * (1) board doesn't support a NIC but the implicit "-net nic"
     * requested one
     * (2) CONFIG_SLIRP not set, in which case the implicit "-net nic"
     * sets up a nic that isn't connected to anything.
     */
    if (!default_net) {
        net_check_clients();
    if (boot_once) {
        qemu_boot_set(boot_once, &error_fatal);
        qemu_register_reset(restore_boot_order, g_strdup(boot_order));
    ds = init_displaystate();
    /* init local displays */
    switch (display_type) {
    case DT_CURSES:
        curses_display_init(ds, full_screen);
    case DT_SDL:
        sdl_display_init(ds, full_screen, no_frame);
    case DT_COCOA:
        cocoa_display_init(ds, full_screen);
    case DT_GTK:
        gtk_display_init(ds, full_screen, grab_on_hover);
    default:
    /* must be after terminal init, SDL library changes signal handlers */
    os_setup_signal_handling();
    /* init remote displays */
#ifdef CONFIG_VNC
    qemu_opts_foreach(qemu_find_opts("vnc"),
                      vnc_init_func, NULL, NULL);
#endif
    if (using_spice) {
        qemu_spice_display_init();
    if (foreach_device_config(DEV_GDB, gdbserver_start) < 0) {
    qdev_machine_creation_done();
    /* TODO: once all bus devices are qdevified, this should be done
     * when bus is created by qdev.c */
    qemu_register_reset(qbus_reset_all_fn, sysbus_get_default());
    qemu_run_machine_init_done_notifiers();
    if (rom_check_and_register_reset() != 0) {
        error_report("rom check and register reset failed");
    replay_start();
    /* This checkpoint is required by replay to separate prior clock
       reading from the other reads, because timer polling functions query
       clock values from the log. */
    replay_checkpoint(CHECKPOINT_RESET);
    qemu_system_reset(VMRESET_SILENT);
    register_global_state();
    if (replay_mode != REPLAY_MODE_NONE) {
        replay_vmstate_init();
    } else if (loadvm) {
        if (load_vmstate(loadvm) < 0) {
            autostart = 0;
    qdev_prop_check_globals();
    if (vmstate_dump_file) {
        /* dump and exit */
        dump_vmstate_json_to_file(vmstate_dump_file);
        return 0;
    if (incoming) {
        Error *local_err = NULL;
        qemu_start_incoming_migration(incoming, &local_err);
        if (local_err) {
            error_reportf_err(local_err, "-incoming %s: ", incoming);
    } else if (autostart) {
        vm_start();
    os_setup_post();
    main_loop();
    replay_disable_events();
    iothread_stop_all();
    bdrv_close_all();
    pause_all_vcpus();
    res_free();
    /* vhost-user must be cleaned up before chardevs.  */
    net_cleanup();
    audio_cleanup();
    monitor_cleanup();
    qemu_chr_cleanup();
    return 0;