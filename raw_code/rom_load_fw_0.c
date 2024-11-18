int rom_load_fw(void *fw_cfg)

{

    Rom *rom;



    QTAILQ_FOREACH(rom, &roms, next) {

        if (!rom->fw_file) {

            continue;

        }

        fw_cfg_add_file(fw_cfg, rom->fw_dir, rom->fw_file, rom->data, rom->romsize);

    }

    return 0;

}
