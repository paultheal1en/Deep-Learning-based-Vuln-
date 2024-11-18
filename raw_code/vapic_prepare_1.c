static int vapic_prepare(VAPICROMState *s)

{

    vapic_map_rom_writable(s);



    if (patch_hypercalls(s) < 0) {

        return -1;

    }



    vapic_enable_tpr_reporting(true);



    return 0;

}
