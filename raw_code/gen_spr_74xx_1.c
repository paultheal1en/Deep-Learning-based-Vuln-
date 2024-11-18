static void gen_spr_74xx (CPUPPCState *env)

{

    /* Processor identification */

    spr_register(env, SPR_PIR, "PIR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_pir,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_MMCR2, "MMCR2",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_UMMCR2, "UMMCR2",

                 &spr_read_ureg, SPR_NOACCESS,

                 &spr_read_ureg, SPR_NOACCESS,

                 0x00000000);

    /* XXX: not implemented */

    spr_register(env, SPR_BAMR, "BAMR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_MSSCR0, "MSSCR0",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

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

    /* Altivec */

    spr_register(env, SPR_VRSAVE, "VRSAVE",

                 &spr_read_generic, &spr_write_generic,

                 &spr_read_generic, &spr_write_generic,

                 0x00000000);

    /* XXX : not implemented */

    spr_register(env, SPR_L2CR, "L2CR",

                 SPR_NOACCESS, SPR_NOACCESS,

                 &spr_read_generic, NULL,

                 0x00000000);

    /* Not strictly an SPR */

    vscr_init(env, 0x00010000);

}
