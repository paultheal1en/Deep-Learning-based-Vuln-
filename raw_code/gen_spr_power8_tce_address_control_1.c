static void gen_spr_power8_tce_address_control(CPUPPCState *env)

{

    spr_register(env, SPR_TAR, "TAR",

                 &spr_read_generic, &spr_write_generic,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

}
