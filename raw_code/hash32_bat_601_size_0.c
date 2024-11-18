static void hash32_bat_601_size(CPUPPCState *env, target_ulong *blp, int *validp,

                                target_ulong batu, target_ulong batl)

{

    target_ulong bl;

    int valid;



    bl = (batl & BATL32_601_BL) << 17;

    LOG_BATS("b %02x ==> bl " TARGET_FMT_lx " msk " TARGET_FMT_lx "\n",

             (uint8_t)(batl & BATL32_601_BL), bl, ~bl);

    valid = !!(batl & BATL32_601_V);

    *blp = bl;

    *validp = valid;

}
