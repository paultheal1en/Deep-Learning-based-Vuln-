static void kzm_init(MachineState *machine)

{

    IMX31KZM *s = g_new0(IMX31KZM, 1);

    unsigned int ram_size;

    unsigned int alias_offset;

    unsigned int i;



    object_initialize(&s->soc, sizeof(s->soc), TYPE_FSL_IMX31);

    object_property_add_child(OBJECT(machine), "soc", OBJECT(&s->soc),

                              &error_abort);



    object_property_set_bool(OBJECT(&s->soc), true, "realized", &error_fatal);



    /* Check the amount of memory is compatible with the SOC */

    if (machine->ram_size > (FSL_IMX31_SDRAM0_SIZE + FSL_IMX31_SDRAM1_SIZE)) {

        error_report("WARNING: RAM size " RAM_ADDR_FMT " above max supported, "

                     "reduced to %x", machine->ram_size,

                     FSL_IMX31_SDRAM0_SIZE + FSL_IMX31_SDRAM1_SIZE);

        machine->ram_size = FSL_IMX31_SDRAM0_SIZE + FSL_IMX31_SDRAM1_SIZE;

    }



    memory_region_allocate_system_memory(&s->ram, NULL, "kzm.ram",

                                         machine->ram_size);

    memory_region_add_subregion(get_system_memory(), FSL_IMX31_SDRAM0_ADDR,

                                &s->ram);



    /* initialize the alias memory if any */

    for (i = 0, ram_size = machine->ram_size, alias_offset = 0;

         (i < 2) && ram_size; i++) {

        unsigned int size;

        static const struct {

            hwaddr addr;

            unsigned int size;

        } ram[2] = {

            { FSL_IMX31_SDRAM0_ADDR, FSL_IMX31_SDRAM0_SIZE },

            { FSL_IMX31_SDRAM1_ADDR, FSL_IMX31_SDRAM1_SIZE },

        };



        size = MIN(ram_size, ram[i].size);



        ram_size -= size;



        if (size < ram[i].size) {

            memory_region_init_alias(&s->ram_alias, NULL, "ram.alias",

                                     &s->ram, alias_offset, ram[i].size - size);

            memory_region_add_subregion(get_system_memory(),

                                        ram[i].addr + size, &s->ram_alias);

        }



        alias_offset += ram[i].size;

    }



    if (nd_table[0].used) {

        lan9118_init(&nd_table[0], KZM_LAN9118_ADDR,

                     qdev_get_gpio_in(DEVICE(&s->soc.avic), 52));

    }



    if (serial_hds[2]) { /* touchscreen */

        serial_mm_init(get_system_memory(), KZM_FPGA_ADDR+0x10, 0,

                       qdev_get_gpio_in(DEVICE(&s->soc.avic), 52),

                       14745600, serial_hds[2], DEVICE_NATIVE_ENDIAN);

    }



    kzm_binfo.ram_size = machine->ram_size;

    kzm_binfo.kernel_filename = machine->kernel_filename;

    kzm_binfo.kernel_cmdline = machine->kernel_cmdline;

    kzm_binfo.initrd_filename = machine->initrd_filename;

    kzm_binfo.nb_cpus = 1;



    if (!qtest_enabled()) {

        arm_load_kernel(&s->soc.cpu, &kzm_binfo);

    }

}
