static void gen_spr_ne_601 (CPUPPCState *env)

{

    /* Exception processing */

    spr_register(env, SPR_DSISR, "DSISR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_DAR, "DAR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* Timer */

    spr_register(env, SPR_DECR, "DECR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_decr, &spr_write_decr,

                 0x00000000);

    /* Memory management */

    spr_register(env, SPR_SDR1, "SDR1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_sdr1, &spr_write_sdr1,

                 0x00000000);

}
