static void init_proc_460 (CPUPPCState *env)

{

    /* Time base */

    gen_tbl(env);

    gen_spr_BookE(env);

    gen_spr_440(env);

    spr_register(env, SPR_BOOKE_MCSR, "MCSR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_BOOKE_MCSRR0, "MCSRR0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_BOOKE_MCSRR1, "MCSRR1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_440_CCR1, "CCR1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_DCRIPR, "SPR_DCRIPR",

                 &spr_read_generic, &spr_write_generic,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* Memory management */

    env->nb_tlb = 64;

    env->nb_ways = 1;

    env->id_tlbs = 0;

    /* XXX: TODO: allocate internal IRQ controller */

}
