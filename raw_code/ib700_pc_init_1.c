static void ib700_pc_init(PCIBus *unused)

{

    register_savevm("ib700_wdt", -1, 0, ib700_save, ib700_load, NULL);



    register_ioport_write(0x441, 2, 1, ib700_write_disable_reg, NULL);

    register_ioport_write(0x443, 2, 1, ib700_write_enable_reg, NULL);

}
