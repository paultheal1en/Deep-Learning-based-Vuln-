static void cg3_realizefn(DeviceState *dev, Error **errp)

{

    SysBusDevice *sbd = SYS_BUS_DEVICE(dev);

    CG3State *s = CG3(dev);

    int ret;

    char *fcode_filename;



    /* FCode ROM */

    vmstate_register_ram_global(&s->rom);

    fcode_filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, CG3_ROM_FILE);

    if (fcode_filename) {

        ret = load_image_targphys(fcode_filename, s->prom_addr,

                                  FCODE_MAX_ROM_SIZE);

        g_free(fcode_filename);

        if (ret < 0 || ret > FCODE_MAX_ROM_SIZE) {

            error_report("cg3: could not load prom '%s'", CG3_ROM_FILE);

        }

    }



    memory_region_init_ram(&s->vram_mem, NULL, "cg3.vram", s->vram_size,

                           &error_abort);

    memory_region_set_log(&s->vram_mem, true, DIRTY_MEMORY_VGA);

    vmstate_register_ram_global(&s->vram_mem);

    sysbus_init_mmio(sbd, &s->vram_mem);



    sysbus_init_irq(sbd, &s->irq);



    s->con = graphic_console_init(DEVICE(dev), 0, &cg3_ops, s);

    qemu_console_resize(s->con, s->width, s->height);

}
