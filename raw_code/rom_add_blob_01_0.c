MemoryRegion *rom_add_blob(const char *name, const void *blob, size_t len,

                   size_t max_len, hwaddr addr, const char *fw_file_name,

                   FWCfgReadCallback fw_callback, void *callback_opaque)

{

    MachineClass *mc = MACHINE_GET_CLASS(qdev_get_machine());

    Rom *rom;

    MemoryRegion *mr = NULL;



    rom           = g_malloc0(sizeof(*rom));

    rom->name     = g_strdup(name);

    rom->addr     = addr;

    rom->romsize  = max_len ? max_len : len;

    rom->datasize = len;

    rom->data     = g_malloc0(rom->datasize);

    memcpy(rom->data, blob, len);

    rom_insert(rom);

    if (fw_file_name && fw_cfg) {

        char devpath[100];

        void *data;



        snprintf(devpath, sizeof(devpath), "/rom@%s", fw_file_name);



        if (mc->rom_file_has_mr) {

            data = rom_set_mr(rom, OBJECT(fw_cfg), devpath);

            mr = rom->mr;

        } else {

            data = rom->data;

        }



        fw_cfg_add_file_callback(fw_cfg, fw_file_name,

                                 fw_callback, callback_opaque,

                                 data, rom->datasize);

    }

    return mr;

}
