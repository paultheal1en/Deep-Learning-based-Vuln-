static void icp_control_init(target_phys_addr_t base)

{

    MemoryRegion *io;



    io = (MemoryRegion *)g_malloc0(sizeof(MemoryRegion));

    memory_region_init_io(io, &icp_control_ops, NULL,

                          "control", 0x00800000);

    memory_region_add_subregion(get_system_memory(), base, io);

    /* ??? Save/restore.  */

}
