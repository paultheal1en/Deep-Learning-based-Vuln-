static void gen_spr_vtb(CPUPPCState *env)

{

    spr_register(env, SPR_VTB, "VTB",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_tbl, SPR_NOACCESS,

                 0x00000000);

}
