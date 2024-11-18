static void init_proc_601 (CPUPPCState *env)

{

    gen_spr_ne_601(env);

    gen_spr_601(env);

    /* Hardware implementation registers */

    /* XXX : not implemented */

    spr_register(env, SPR_HID0, "HID0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_HID1, "HID1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_601_HID2, "HID2",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_601_HID5, "HID5",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_601_HID15, "HID15",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* Memory management */

#if !defined(CONFIG_USER_ONLY)

    env->nb_tlb = 64;

    env->nb_ways = 2;

    env->id_tlbs = 0;

#endif

    init_excp_601(env);

    env->dcache_line_size = 64;

    env->icache_line_size = 64;

    /* XXX: TODO: allocate internal IRQ controller */

}
