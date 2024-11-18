static void gen_spr_970_pmu_sup(CPUPPCState *env)

{

    spr_register(env, SPR_970_PMC7, "PMC7",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_970_PMC8, "PMC8",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

}
