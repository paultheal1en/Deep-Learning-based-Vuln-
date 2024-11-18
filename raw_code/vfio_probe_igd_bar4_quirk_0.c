static void vfio_probe_igd_bar4_quirk(VFIOPCIDevice *vdev, int nr)

{

    struct vfio_region_info *rom = NULL, *opregion = NULL,

                            *host = NULL, *lpc = NULL;

    VFIOQuirk *quirk;

    VFIOIGDQuirk *igd;

    PCIDevice *lpc_bridge;

    int i, ret, ggms_mb, gms_mb = 0, gen;

    uint64_t *bdsm_size;

    uint32_t gmch;

    uint16_t cmd_orig, cmd;

    Error *err = NULL;



    /* This must be an Intel VGA device. */

    if (!vfio_pci_is(vdev, PCI_VENDOR_ID_INTEL, PCI_ANY_ID) ||

        !vfio_is_vga(vdev) || nr != 4) {

        return;

    }



    /*

     * IGD is not a standard, they like to change their specs often.  We

     * only attempt to support back to SandBridge and we hope that newer

     * devices maintain compatibility with generation 8.

     */

    gen = igd_gen(vdev);

    if (gen != 6 && gen != 8) {

        error_report("IGD device %s is unsupported by IGD quirks, "

                     "try SandyBridge or newer", vdev->vbasedev.name);

        return;

    }



    /*

     * Regardless of running in UPT or legacy mode, the guest graphics

     * driver may attempt to use stolen memory, however only legacy mode

     * has BIOS support for reserving stolen memory in the guest VM.

     * Emulate the GMCH register in all cases and zero out the stolen

     * memory size here. Legacy mode may request allocation and re-write

     * this below.

     */

    gmch = vfio_pci_read_config(&vdev->pdev, IGD_GMCH, 4);

    gmch &= ~((gen < 8 ? 0x1f : 0xff) << (gen < 8 ? 3 : 8));



    /* GMCH is read-only, emulated */

    pci_set_long(vdev->pdev.config + IGD_GMCH, gmch);

    pci_set_long(vdev->pdev.wmask + IGD_GMCH, 0);

    pci_set_long(vdev->emulated_config_bits + IGD_GMCH, ~0);



    /*

     * This must be at address 00:02.0 for us to even onsider enabling

     * legacy mode.  The vBIOS has dependencies on the PCI bus address.

     */

    if (&vdev->pdev != pci_find_device(pci_device_root_bus(&vdev->pdev),

                                       0, PCI_DEVFN(0x2, 0))) {

        return;

    }



    /*

     * We need to create an LPC/ISA bridge at PCI bus address 00:1f.0 that we

     * can stuff host values into, so if there's already one there and it's not

     * one we can hack on, legacy mode is no-go.  Sorry Q35.

     */

    lpc_bridge = pci_find_device(pci_device_root_bus(&vdev->pdev),

                                 0, PCI_DEVFN(0x1f, 0));

    if (lpc_bridge && !object_dynamic_cast(OBJECT(lpc_bridge),

                                           "vfio-pci-igd-lpc-bridge")) {

        error_report("IGD device %s cannot support legacy mode due to existing "

                     "devices at address 1f.0", vdev->vbasedev.name);

        return;

    }



    /*

     * Most of what we're doing here is to enable the ROM to run, so if

     * there's no ROM, there's no point in setting up this quirk.

     * NB. We only seem to get BIOS ROMs, so a UEFI VM would need CSM support.

     */

    ret = vfio_get_region_info(&vdev->vbasedev,

                               VFIO_PCI_ROM_REGION_INDEX, &rom);

    if ((ret || !rom->size) && !vdev->pdev.romfile) {

        error_report("IGD device %s has no ROM, legacy mode disabled",

                     vdev->vbasedev.name);

        goto out;

    }



    /*

     * Ignore the hotplug corner case, mark the ROM failed, we can't

     * create the devices we need for legacy mode in the hotplug scenario.

     */

    if (vdev->pdev.qdev.hotplugged) {

        error_report("IGD device %s hotplugged, ROM disabled, "

                     "legacy mode disabled", vdev->vbasedev.name);

        vdev->rom_read_failed = true;

        goto out;

    }



    /*

     * Check whether we have all the vfio device specific regions to

     * support legacy mode (added in Linux v4.6).  If not, bail.

     */

    ret = vfio_get_dev_region_info(&vdev->vbasedev,

                        VFIO_REGION_TYPE_PCI_VENDOR_TYPE | PCI_VENDOR_ID_INTEL,

                        VFIO_REGION_SUBTYPE_INTEL_IGD_OPREGION, &opregion);

    if (ret) {

        error_report("IGD device %s does not support OpRegion access,"

                     "legacy mode disabled", vdev->vbasedev.name);

        goto out;

    }



    ret = vfio_get_dev_region_info(&vdev->vbasedev,

                        VFIO_REGION_TYPE_PCI_VENDOR_TYPE | PCI_VENDOR_ID_INTEL,

                        VFIO_REGION_SUBTYPE_INTEL_IGD_HOST_CFG, &host);

    if (ret) {

        error_report("IGD device %s does not support host bridge access,"

                     "legacy mode disabled", vdev->vbasedev.name);

        goto out;

    }



    ret = vfio_get_dev_region_info(&vdev->vbasedev,

                        VFIO_REGION_TYPE_PCI_VENDOR_TYPE | PCI_VENDOR_ID_INTEL,

                        VFIO_REGION_SUBTYPE_INTEL_IGD_LPC_CFG, &lpc);

    if (ret) {

        error_report("IGD device %s does not support LPC bridge access,"

                     "legacy mode disabled", vdev->vbasedev.name);

        goto out;

    }



    /*

     * If IGD VGA Disable is clear (expected) and VGA is not already enabled,

     * try to enable it.  Probably shouldn't be using legacy mode without VGA,

     * but also no point in us enabling VGA if disabled in hardware.

     */

    if (!(gmch & 0x2) && !vdev->vga && vfio_populate_vga(vdev, &err)) {

        error_reportf_err(err, ERR_PREFIX, vdev->vbasedev.name);

        error_report("IGD device %s failed to enable VGA access, "

                     "legacy mode disabled", vdev->vbasedev.name);

        goto out;

    }



    /* Create our LPC/ISA bridge */

    ret = vfio_pci_igd_lpc_init(vdev, lpc);

    if (ret) {

        error_report("IGD device %s failed to create LPC bridge, "

                     "legacy mode disabled", vdev->vbasedev.name);

        goto out;

    }



    /* Stuff some host values into the VM PCI host bridge */

    ret = vfio_pci_igd_host_init(vdev, host);

    if (ret) {

        error_report("IGD device %s failed to modify host bridge, "

                     "legacy mode disabled", vdev->vbasedev.name);

        goto out;

    }



    /* Setup OpRegion access */

    ret = vfio_pci_igd_opregion_init(vdev, opregion, &err);

    if (ret) {

        error_append_hint(&err, "IGD legacy mode disabled\n");

        error_reportf_err(err, ERR_PREFIX, vdev->vbasedev.name);

        goto out;

    }



    /* Setup our quirk to munge GTT addresses to the VM allocated buffer */

    quirk = g_malloc0(sizeof(*quirk));

    quirk->mem = g_new0(MemoryRegion, 2);

    quirk->nr_mem = 2;

    igd = quirk->data = g_malloc0(sizeof(*igd));

    igd->vdev = vdev;

    igd->index = ~0;

    igd->bdsm = vfio_pci_read_config(&vdev->pdev, IGD_BDSM, 4);

    igd->bdsm &= ~((1 << 20) - 1); /* 1MB aligned */



    memory_region_init_io(&quirk->mem[0], OBJECT(vdev), &vfio_igd_index_quirk,

                          igd, "vfio-igd-index-quirk", 4);

    memory_region_add_subregion_overlap(vdev->bars[nr].region.mem,

                                        0, &quirk->mem[0], 1);



    memory_region_init_io(&quirk->mem[1], OBJECT(vdev), &vfio_igd_data_quirk,

                          igd, "vfio-igd-data-quirk", 4);

    memory_region_add_subregion_overlap(vdev->bars[nr].region.mem,

                                        4, &quirk->mem[1], 1);



    QLIST_INSERT_HEAD(&vdev->bars[nr].quirks, quirk, next);



    /* Determine the size of stolen memory needed for GTT */

    ggms_mb = (gmch >> (gen < 8 ? 8 : 6)) & 0x3;

    if (gen > 6) {

        ggms_mb = 1 << ggms_mb;

    }



    /*

     * Assume we have no GMS memory, but allow it to be overrided by device

     * option (experimental).  The spec doesn't actually allow zero GMS when

     * when IVD (IGD VGA Disable) is clear, but the claim is that it's unused,

     * so let's not waste VM memory for it.

     */

    if (vdev->igd_gms) {

        if (vdev->igd_gms <= 0x10) {

            gms_mb = vdev->igd_gms * 32;

            gmch |= vdev->igd_gms << (gen < 8 ? 3 : 8);

            pci_set_long(vdev->pdev.config + IGD_GMCH, gmch);

        } else {

            error_report("Unsupported IGD GMS value 0x%x", vdev->igd_gms);

            vdev->igd_gms = 0;

        }

    }



    /*

     * Request reserved memory for stolen memory via fw_cfg.  VM firmware

     * must allocate a 1MB aligned reserved memory region below 4GB with

     * the requested size (in bytes) for use by the Intel PCI class VGA

     * device at VM address 00:02.0.  The base address of this reserved

     * memory region must be written to the device BDSM regsiter at PCI

     * config offset 0x5C.

     */

    bdsm_size = g_malloc(sizeof(*bdsm_size));

    *bdsm_size = cpu_to_le64((ggms_mb + gms_mb) * 1024 * 1024);

    fw_cfg_add_file(fw_cfg_find(), "etc/igd-bdsm-size",

                    bdsm_size, sizeof(*bdsm_size));



    /* BDSM is read-write, emulated.  The BIOS needs to be able to write it */

    pci_set_long(vdev->pdev.config + IGD_BDSM, 0);

    pci_set_long(vdev->pdev.wmask + IGD_BDSM, ~0);

    pci_set_long(vdev->emulated_config_bits + IGD_BDSM, ~0);



    /*

     * This IOBAR gives us access to GTTADR, which allows us to write to

     * the GTT itself.  So let's go ahead and write zero to all the GTT

     * entries to avoid spurious DMA faults.  Be sure I/O access is enabled

     * before talking to the device.

     */

    if (pread(vdev->vbasedev.fd, &cmd_orig, sizeof(cmd_orig),

              vdev->config_offset + PCI_COMMAND) != sizeof(cmd_orig)) {

        error_report("IGD device %s - failed to read PCI command register",

                     vdev->vbasedev.name);

    }



    cmd = cmd_orig | PCI_COMMAND_IO;



    if (pwrite(vdev->vbasedev.fd, &cmd, sizeof(cmd),

               vdev->config_offset + PCI_COMMAND) != sizeof(cmd)) {

        error_report("IGD device %s - failed to write PCI command register",

                     vdev->vbasedev.name);

    }



    for (i = 1; i < vfio_igd_gtt_max(vdev); i += 4) {

        vfio_region_write(&vdev->bars[4].region, 0, i, 4);

        vfio_region_write(&vdev->bars[4].region, 4, 0, 4);

    }



    if (pwrite(vdev->vbasedev.fd, &cmd_orig, sizeof(cmd_orig),

               vdev->config_offset + PCI_COMMAND) != sizeof(cmd_orig)) {

        error_report("IGD device %s - failed to restore PCI command register",

                     vdev->vbasedev.name);

    }



    trace_vfio_pci_igd_bdsm_enabled(vdev->vbasedev.name, ggms_mb + gms_mb);



out:

    g_free(rom);

    g_free(opregion);

    g_free(host);

    g_free(lpc);

}
