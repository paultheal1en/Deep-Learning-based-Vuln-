static void gen_spr_403 (CPUPPCState *env)

{

    /* MMU */

    spr_register(env, SPR_403_PBL1,  "PBL1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_403_pbr, &spr_write_403_pbr,

                 0x00000000);

    spr_register(env, SPR_403_PBU1,  "PBU1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_403_pbr, &spr_write_403_pbr,

                 0x00000000);

    spr_register(env, SPR_403_PBL2,  "PBL2",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_403_pbr, &spr_write_403_pbr,

                 0x00000000);

    spr_register(env, SPR_403_PBU2,  "PBU2",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_403_pbr, &spr_write_403_pbr,

                 0x00000000);

    /* Debug */

    /* XXX : not implemented */

    spr_register(env, SPR_40x_DAC2, "DAC2",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_40x_IAC2, "IAC2",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

}
