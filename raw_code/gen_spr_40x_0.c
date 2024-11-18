static void gen_spr_40x (CPUPPCState *env)

{

    /* Cache */

    /* XXX : not implemented */

    spr_register(env, SPR_40x_DCCR, "DCCR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_40x_DCWR, "DCWR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_40x_ICCR, "ICCR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_BOOKE_ICBDR, "ICBDR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, SPR_NOACCESS,

                 0x00000000);

    /* Bus access control */

    spr_register(env, SPR_40x_SGR, "SGR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0xFFFFFFFF);

    spr_register(env, SPR_40x_ZPR, "ZPR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* MMU */

    spr_register(env, SPR_40x_PID, "PID",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* Exception */

    spr_register(env, SPR_40x_DEAR, "DEAR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_40x_ESR, "ESR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_40x_EVPR, "EVPR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_40x_SRR2, "SRR2",

                 &spr_read_generic, &spr_write_generic,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_40x_SRR3, "SRR3",

                 &spr_read_generic, &spr_write_generic,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* Timers */

    spr_register(env, SPR_40x_PIT, "PIT",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_40x_pit, &spr_write_40x_pit,

                 0x00000000);

    spr_register(env, SPR_40x_TCR, "TCR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_booke_tcr,

                 0x00000000);

    spr_register(env, SPR_40x_TSR, "TSR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_booke_tsr,

                 0x00000000);

    /* Debug interface */

    /* XXX : not implemented */

    spr_register(env, SPR_40x_DAC1, "DAC1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_40x_DAC2, "DAC2",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_40x_DBCR0, "DBCR0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_40x_dbcr0,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_40x_DBSR, "DBSR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_clear,

                 /* Last reset was system reset */

                 0x00000300);

    /* XXX : not implemented */

    spr_register(env, SPR_40x_IAC1, "IAC1",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    spr_register(env, SPR_40x_IAC2, "IAC2",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

}
