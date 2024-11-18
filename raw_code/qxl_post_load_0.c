static int qxl_post_load(void *opaque, int version)

{

    PCIQXLDevice* d = opaque;

    uint8_t *ram_start = d->vga.vram_ptr;

    QXLCommandExt *cmds;

    int in, out, i, newmode;



    dprint(d, 1, "%s: start\n", __FUNCTION__);



    assert(d->last_release_offset < d->vga.vram_size);

    if (d->last_release_offset == 0) {

        d->last_release = NULL;

    } else {

        d->last_release = (QXLReleaseInfo *)(ram_start + d->last_release_offset);

    }



    d->modes = (QXLModes*)((uint8_t*)d->rom + d->rom->modes_offset);



    dprint(d, 1, "%s: restore mode (%s)\n", __FUNCTION__,

        qxl_mode_to_string(d->mode));

    newmode = d->mode;

    d->mode = QXL_MODE_UNDEFINED;

    switch (newmode) {

    case QXL_MODE_UNDEFINED:

        break;

    case QXL_MODE_VGA:

        qxl_enter_vga_mode(d);

        break;

    case QXL_MODE_NATIVE:

        for (i = 0; i < NUM_MEMSLOTS; i++) {

            if (!d->guest_slots[i].active) {

                continue;

            }

            qxl_add_memslot(d, i, 0, QXL_SYNC);

        }

        qxl_create_guest_primary(d, 1, QXL_SYNC);



        /* replay surface-create and cursor-set commands */

        cmds = g_malloc0(sizeof(QXLCommandExt) * (NUM_SURFACES + 1));

        for (in = 0, out = 0; in < NUM_SURFACES; in++) {

            if (d->guest_surfaces.cmds[in] == 0) {

                continue;

            }

            cmds[out].cmd.data = d->guest_surfaces.cmds[in];

            cmds[out].cmd.type = QXL_CMD_SURFACE;

            cmds[out].group_id = MEMSLOT_GROUP_GUEST;

            out++;

        }

        cmds[out].cmd.data = d->guest_cursor;

        cmds[out].cmd.type = QXL_CMD_CURSOR;

        cmds[out].group_id = MEMSLOT_GROUP_GUEST;

        out++;

        qxl_spice_loadvm_commands(d, cmds, out);

        g_free(cmds);



        break;

    case QXL_MODE_COMPAT:

        qxl_set_mode(d, d->shadow_rom.mode, 1);

        break;

    }

    dprint(d, 1, "%s: done\n", __FUNCTION__);



    return 0;

}
