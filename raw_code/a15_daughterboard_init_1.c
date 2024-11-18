static void a15_daughterboard_init(const VexpressMachineState *vms,

                                   ram_addr_t ram_size,

                                   const char *cpu_model,

                                   qemu_irq *pic)

{

    MemoryRegion *sysmem = get_system_memory();

    MemoryRegion *ram = g_new(MemoryRegion, 1);

    MemoryRegion *sram = g_new(MemoryRegion, 1);



    if (!cpu_model) {

        cpu_model = "cortex-a15";

    }



    {

        /* We have to use a separate 64 bit variable here to avoid the gcc

         * "comparison is always false due to limited range of data type"

         * warning if we are on a host where ram_addr_t is 32 bits.

         */

        uint64_t rsz = ram_size;

        if (rsz > (30ULL * 1024 * 1024 * 1024)) {

            fprintf(stderr, "vexpress-a15: cannot model more than 30GB RAM\n");

            exit(1);

        }

    }



    memory_region_allocate_system_memory(ram, NULL, "vexpress.highmem",

                                         ram_size);

    /* RAM is from 0x80000000 upwards; there is no low-memory alias for it. */

    memory_region_add_subregion(sysmem, 0x80000000, ram);



    /* 0x2c000000 A15MPCore private memory region (GIC) */

    init_cpus(cpu_model, "a15mpcore_priv", 0x2c000000, pic, vms->secure);



    /* A15 daughterboard peripherals: */



    /* 0x20000000: CoreSight interfaces: not modelled */

    /* 0x2a000000: PL301 AXI interconnect: not modelled */

    /* 0x2a420000: SCC: not modelled */

    /* 0x2a430000: system counter: not modelled */

    /* 0x2b000000: HDLCD controller: not modelled */

    /* 0x2b060000: SP805 watchdog: not modelled */

    /* 0x2b0a0000: PL341 dynamic memory controller: not modelled */

    /* 0x2e000000: system SRAM */

    memory_region_init_ram(sram, NULL, "vexpress.a15sram", 0x10000,

                           &error_abort);

    vmstate_register_ram_global(sram);

    memory_region_add_subregion(sysmem, 0x2e000000, sram);



    /* 0x7ffb0000: DMA330 DMA controller: not modelled */

    /* 0x7ffd0000: PL354 static memory controller: not modelled */

}
