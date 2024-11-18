static void pc_cmos_init_late(void *opaque)

{

    pc_cmos_init_late_arg *arg = opaque;

    ISADevice *s = arg->rtc_state;

    int16_t cylinders;

    int8_t heads, sectors;

    int val;

    int i, trans;

    Object *container;

    CheckFdcState state = { 0 };



    val = 0;

    if (ide_get_geometry(arg->idebus[0], 0,

                         &cylinders, &heads, &sectors) >= 0) {

        cmos_init_hd(s, 0x19, 0x1b, cylinders, heads, sectors);

        val |= 0xf0;

    }

    if (ide_get_geometry(arg->idebus[0], 1,

                         &cylinders, &heads, &sectors) >= 0) {

        cmos_init_hd(s, 0x1a, 0x24, cylinders, heads, sectors);

        val |= 0x0f;

    }

    rtc_set_memory(s, 0x12, val);



    val = 0;

    for (i = 0; i < 4; i++) {

        /* NOTE: ide_get_geometry() returns the physical

           geometry.  It is always such that: 1 <= sects <= 63, 1

           <= heads <= 16, 1 <= cylinders <= 16383. The BIOS

           geometry can be different if a translation is done. */

        if (ide_get_geometry(arg->idebus[i / 2], i % 2,

                             &cylinders, &heads, &sectors) >= 0) {

            trans = ide_get_bios_chs_trans(arg->idebus[i / 2], i % 2) - 1;

            assert((trans & ~3) == 0);

            val |= trans << (i * 2);

        }

    }

    rtc_set_memory(s, 0x39, val);



    /*

     * Locate the FDC at IO address 0x3f0, and configure the CMOS registers

     * accordingly.

     */

    for (i = 0; i < ARRAY_SIZE(fdc_container_path); i++) {

        container = container_get(qdev_get_machine(), fdc_container_path[i]);

        object_child_foreach(container, check_fdc, &state);

    }



    if (state.multiple) {

        error_report("warning: multiple floppy disk controllers with "

                     "iobase=0x3f0 have been found;\n"

                     "the one being picked for CMOS setup might not reflect "

                     "your intent");

    }

    pc_cmos_init_floppy(s, state.floppy);



    qemu_unregister_reset(pc_cmos_init_late, opaque);

}
